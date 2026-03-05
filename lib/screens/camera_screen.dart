import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/detection.dart';
import '../models/scan_entry.dart';
import '../services/api_service.dart';
import '../services/camera_service.dart';
import '../services/export_service.dart';
import '../services/frame_quality_service.dart';
import '../services/tracking_service.dart';
import '../widgets/bounding_box_painter.dart';
import '../widgets/scan_line_painter.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with SingleTickerProviderStateMixin {
  final CameraService _cameraService = CameraService();
  late final ApiService _apiService;
  late final TrackingService _trackingService;
  final FrameQualityService _frameQualityService = FrameQualityService();
  final ExportService _exportService = createExportService();

  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isProcessing = false;
  bool _isDecoding = false;
  bool _torchOn = false;
  bool _showSuccessFlash = false;
  double _currentZoom = 1.0;
  Timer? _scanTimer;
  Timer? _flashTimer;
  Timer? _bannerTimer;
  DateTime? _lastDecodeRequestAt;

  Duration _currentInterval = const Duration(milliseconds: 800);

  List<Detection> _detections = [];
  String? _decodedBannerText;
  String? _errorMessage;
  final List<ScanEntry> _scanHistory = [];
  final Set<String> _scannedBarcodes = {};
  final List<XFile> _frameBuffer = [];

  static const Rect _roiNormalized = Rect.fromLTWH(0.2, 0.3, 0.6, 0.4);
  static const Duration _decodeCooldown = Duration(milliseconds: 700);

  late final AnimationController _scanLineController;

  @override
  void initState() {
    super.initState();
    _trackingService = TrackingService(roiNormalized: _roiNormalized);
    _apiService = ApiService(roiNormalized: _roiNormalized);
    _scanLineController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      final controller = await _cameraService.initializeCamera();
      if (!mounted) return;
      setState(() {
        _cameraController = controller;
        _isCameraInitialized = true;
      });
      _startScanTimer();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Failed to initialize camera: $e';
      });
    }
  }

  Future<void> _captureAndDetect() async {
    final controller = _cameraController;
    if (!_isCameraInitialized || controller == null || _isProcessing) {
      return;
    }

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      final XFile imageFile = await controller.takePicture();
      final Uint8List bytes = await imageFile.readAsBytes();
      final quality = _frameQualityService.computeQualityScore(bytes);

      _frameBuffer.add(imageFile);
      if (_frameBuffer.length > 8) {
        _frameBuffer.removeAt(0);
      }

      if (!_canSendDecodeRequest()) {
        _updateScanInterval();
        return;
      }

      final hasVisibleDetection = _detections.any(_isInsideRoiDetection);
      final framesToSend = _takeLatestFrames(
        _frameBuffer,
        maxCount: hasVisibleDetection ? 8 : 4,
      );

      _lastDecodeRequestAt = DateTime.now();
      _isDecoding = true;
      final detections = await _apiService.detectBarcodes(framesToSend);
      _isDecoding = false;

      debugPrint(
        'Decode stats: latency=${_apiService.lastLatencyMs}ms '
        'frames=${_apiService.lastFramesSent} '
        'decoded=${_apiService.lastDecodedValue ?? '-'} '
        'strategy=${_apiService.lastStrategy ?? '-'}',
      );

      final stabilizedDetections = _trackingService.trackAndFilter(
        detections: detections,
        frame: imageFile,
        sharpness: quality,
        minFramesForDecode: 6,
        sharpnessThreshold: 80,
      );

      String? decoded;
      if (stabilizedDetections.isNotEmpty) {
        decoded = stabilizedDetections.first.decoded;
      }

      await _applyAutoZoom(controller, stabilizedDetections);

      if (!mounted) return;
      setState(() {
        _detections = stabilizedDetections;
      });

      if (decoded != null && decoded.isNotEmpty) {
        _handleDecodedBarcode(decoded, barcodeType: _apiService.lastDecodedType);
      }

      _updateScanInterval();
    } catch (e) {
      _isDecoding = false;
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Error during capture/detect: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  bool _canSendDecodeRequest() {
    if (_isDecoding) return false;

    final lastAt = _lastDecodeRequestAt;
    if (lastAt == null) return true;

    final elapsed = DateTime.now().difference(lastAt);
    return elapsed >= _decodeCooldown;
  }

  bool _isInsideRoiDetection(Detection d) {
    final cx = (d.x1 + d.x2) / 2;
    final cy = (d.y1 + d.y2) / 2;
    return cx >= _roiNormalized.left &&
        cx <= _roiNormalized.right &&
        cy >= _roiNormalized.top &&
        cy <= _roiNormalized.bottom;
  }

  List<XFile> _takeLatestFrames(List<XFile> frames, {required int maxCount}) {
    if (frames.length <= maxCount) {
      return List<XFile>.from(frames);
    }
    return List<XFile>.from(frames.sublist(frames.length - maxCount));
  }

  @override
  void dispose() {
    _scanTimer?.cancel();
    _flashTimer?.cancel();
    _bannerTimer?.cancel();
    _scanLineController.dispose();
    _cameraController?.dispose();
    super.dispose();
  }

  void _startScanTimer() {
    _scanTimer?.cancel();
    _scanTimer = Timer.periodic(_currentInterval, (_) => _captureAndDetect());
  }

  void _updateScanInterval() {
    bool hasCandidate = false;
    bool hasStable = false;
    for (final track in _trackingService.tracks) {
      if (track.framesSeen >= 8) {
        hasStable = true;
        break;
      }
      if (track.framesSeen >= 4) {
        hasCandidate = true;
      }
    }

    Duration desired;
    if (hasStable) {
      desired = const Duration(milliseconds: 250);
    } else if (hasCandidate) {
      desired = const Duration(milliseconds: 450);
    } else {
      desired = const Duration(milliseconds: 800);
    }

    if (desired == _currentInterval) return;
    _currentInterval = desired;
    _startScanTimer();
  }

  void _handleDecodedBarcode(String decoded, {String? barcodeType}) {
    if (_scannedBarcodes.contains(decoded)) {
      return;
    }

    _scannedBarcodes.add(decoded);
    final now = DateTime.now();
    _scanHistory.insert(
      0,
      ScanEntry(
        barcode: decoded,
        timestamp: now,
        barcodeType: barcodeType,
      ),
    );

    if (_scanHistory.length > 100) {
      _scanHistory.removeRange(100, _scanHistory.length);
    }

    setState(() {
      _decodedBannerText = decoded;
      _showSuccessFlash = true;
    });

    _flashTimer?.cancel();
    _flashTimer = Timer(const Duration(milliseconds: 200), () {
      if (!mounted) return;
      setState(() {
        _showSuccessFlash = false;
      });
    });

    _bannerTimer?.cancel();
    _bannerTimer = Timer(const Duration(seconds: 2), () {
      if (!mounted) return;
      setState(() {
        _decodedBannerText = null;
      });
    });

    _playFeedback();
  }

  Future<void> _exportCsv() async {
    if (_scanHistory.isEmpty) {
      setState(() {
        _errorMessage = 'Nothing to export yet.';
      });
      return;
    }

    final buffer = StringBuffer('barcode,timestamp,type\n');
    for (final entry in _scanHistory) {
      final ts = _formatTimestamp(entry.timestamp);
      final type = entry.barcodeType ?? '';
      buffer.writeln('${entry.barcode},$ts,$type');
    }

    try {
      await _exportService.exportCsv(buffer.toString());
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Failed to export CSV: $e';
      });
    }
  }

  String _formatTimestamp(DateTime dt) {
    String two(int n) => n.toString().padLeft(2, '0');
    final y = dt.year.toString().padLeft(4, '0');
    final m = two(dt.month);
    final d = two(dt.day);
    final h = two(dt.hour);
    final min = two(dt.minute);
    return '$y-$m-$d $h:$min';
  }

  void _playFeedback() {
    SystemSound.play(SystemSoundType.click);
    HapticFeedback.mediumImpact();
  }

  Future<void> _toggleTorch() async {
    final controller = _cameraController;
    if (controller == null || !_isCameraInitialized) return;
    try {
      _torchOn = !_torchOn;
      await controller.setFlashMode(_torchOn ? FlashMode.torch : FlashMode.off);
      setState(() {});
    } catch (_) {}
  }

  void _showComingSoon(String label) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('$label will be available soon.'),
        duration: const Duration(milliseconds: 1200),
      ),
    );
  }

  Future<void> _applyAutoZoom(
    CameraController controller,
    List<Detection> detections,
  ) async {
    if (detections.isEmpty) {
      if (_currentZoom != 1.0) {
        _currentZoom = 1.0;
        await controller.setZoomLevel(_currentZoom);
      }
      return;
    }

    final d = detections.first;
    final width = (d.x2 - d.x1).clamp(0.0, 1.0);
    final height = (d.y2 - d.y1).clamp(0.0, 1.0);
    final area = width * height;

    double targetZoom = _currentZoom;
    if (area < 0.15) {
      targetZoom = (_currentZoom + 0.5).clamp(1.0, 4.0);
    } else if (area > 0.35) {
      targetZoom = 1.0;
    }

    final smoothed = 0.8 * _currentZoom + 0.2 * targetZoom;
    final clamped = smoothed.clamp(1.0, 4.0).toDouble();

    if ((clamped - _currentZoom).abs() > 0.02) {
      _currentZoom = clamped;
      await controller.setZoomLevel(_currentZoom);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF050607),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final isWide = constraints.maxWidth >= 980;

            if (isWide) {
              return Row(
                children: [
                  Expanded(
                    child: Center(
                      child: Padding(
                        padding: const EdgeInsets.all(24),
                        child: ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 900),
                          child: _buildScannerStage(),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(
                    width: 360,
                    child: _buildHistoryPanel(),
                  ),
                ],
              );
            }

            return Column(
              children: [
                Expanded(
                  child: _buildScannerStage(),
                ),
                SizedBox(
                  height: 260,
                  child: _buildHistoryPanel(),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _buildScannerStage() {
    final controller = _cameraController;

    if (_errorMessage != null) {
      return Center(
        child: Text(
          _errorMessage!,
          style: const TextStyle(color: Colors.redAccent),
          textAlign: TextAlign.center,
        ),
      );
    }

    if (!_isCameraInitialized ||
        controller == null ||
        !controller.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.greenAccent),
      );
    }

    return Container(
      margin: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white12),
        boxShadow: const [
          BoxShadow(
            color: Color(0x55000000),
            blurRadius: 20,
            offset: Offset(0, 8),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(controller),
            CustomPaint(
              painter: _ScannerOverlayPainter(_roiNormalized),
            ),
            AnimatedBuilder(
              animation: _scanLineController,
              builder: (context, _) {
                return CustomPaint(
                  painter: ScanLinePainter(
                    progress: _scanLineController.value,
                    roi: _roiNormalized,
                  ),
                );
              },
            ),
            if (_detections.isNotEmpty)
              CustomPaint(
                painter: BoundingBoxPainter(_detections),
              ),
            if (_showSuccessFlash)
              IgnorePointer(
                child: AnimatedOpacity(
                  opacity: _showSuccessFlash ? 1.0 : 0.0,
                  duration: const Duration(milliseconds: 140),
                  child: Container(color: Colors.greenAccent.withOpacity(0.18)),
                ),
              ),
            Positioned(
              top: 12,
              left: 12,
              right: 12,
              child: _buildTopBar(),
            ),
            Positioned(
              left: 12,
              right: 12,
              bottom: 12,
              child: _buildBottomPanel(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xAA0C0E10),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white12),
      ),
      child: Row(
        children: [
          const Text(
            'Barcode AI Scanner',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.2,
            ),
          ),
          const Spacer(),
          IconButton(
            onPressed: _toggleTorch,
            icon: Icon(
              _torchOn ? Icons.flash_on_rounded : Icons.flash_off_rounded,
              color: _torchOn ? Colors.greenAccent : Colors.white70,
            ),
            tooltip: 'Flashlight',
          ),
          IconButton(
            onPressed: () => _showComingSoon('Camera switch'),
            icon: const Icon(Icons.cameraswitch_rounded, color: Colors.white70),
            tooltip: 'Switch camera',
          ),
          IconButton(
            onPressed: () => _showComingSoon('Settings'),
            icon: const Icon(Icons.tune_rounded, color: Colors.white70),
            tooltip: 'Settings',
          ),
        ],
      ),
    );
  }

  Widget _buildBottomPanel() {
    final status = _decodedBannerText ?? 'No barcode detected yet.';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xB0101215),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white12),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              status,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: _decodedBannerText == null ? Colors.white70 : Colors.greenAccent,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          const SizedBox(width: 12),
          FilledButton.icon(
            onPressed: _exportCsv,
            icon: const Icon(Icons.file_upload_outlined, size: 18),
            label: const Text('Export CSV'),
            style: FilledButton.styleFrom(
              backgroundColor: const Color(0xFF1C8F57),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHistoryPanel() {
    final items = _scanHistory.take(10).toList();

    return Container(
      margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      decoration: BoxDecoration(
        color: const Color(0xFF121417),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(Icons.history_rounded, color: Colors.greenAccent, size: 18),
              SizedBox(width: 8),
              Text(
                'Scan History',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Expanded(
            child: items.isEmpty
                ? const Center(
                    child: Text(
                      'No scans yet.',
                      style: TextStyle(color: Colors.white54),
                    ),
                  )
                : ListView.separated(
                    itemCount: items.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 8),
                    itemBuilder: (context, index) {
                      final entry = items[index];
                      final type = entry.barcodeType ?? 'UNKNOWN';

                      return Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 10,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1B1E22),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.white10),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              entry.barcode,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                color: Colors.greenAccent,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              _formatTimestamp(entry.timestamp),
                              style: const TextStyle(color: Colors.white70, fontSize: 12),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'Type: $type',
                              style: const TextStyle(color: Colors.white60, fontSize: 12),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _ScannerOverlayPainter extends CustomPainter {
  final Rect roiNormalized;

  _ScannerOverlayPainter(this.roiNormalized);

  @override
  void paint(Canvas canvas, Size size) {
    final roi = Rect.fromLTWH(
      roiNormalized.left * size.width,
      roiNormalized.top * size.height,
      roiNormalized.width * size.width,
      roiNormalized.height * size.height,
    );

    final full = Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height));
    final window = Path()
      ..addRRect(
        RRect.fromRectAndRadius(roi, const Radius.circular(16)),
      );
    final mask = Path.combine(PathOperation.difference, full, window);

    canvas.drawPath(mask, Paint()..color = Colors.black.withOpacity(0.45));

    final border = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.white.withOpacity(0.75);
    canvas.drawRRect(
      RRect.fromRectAndRadius(roi, const Radius.circular(16)),
      border,
    );

    final corner = Paint()
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 4
      ..color = Colors.greenAccent;

    const cornerLen = 24.0;

    canvas.drawLine(Offset(roi.left, roi.top + cornerLen), Offset(roi.left, roi.top), corner);
    canvas.drawLine(Offset(roi.left, roi.top), Offset(roi.left + cornerLen, roi.top), corner);

    canvas.drawLine(Offset(roi.right - cornerLen, roi.top), Offset(roi.right, roi.top), corner);
    canvas.drawLine(Offset(roi.right, roi.top), Offset(roi.right, roi.top + cornerLen), corner);

    canvas.drawLine(
      Offset(roi.left, roi.bottom - cornerLen),
      Offset(roi.left, roi.bottom),
      corner,
    );
    canvas.drawLine(Offset(roi.left, roi.bottom), Offset(roi.left + cornerLen, roi.bottom), corner);

    canvas.drawLine(
      Offset(roi.right - cornerLen, roi.bottom),
      Offset(roi.right, roi.bottom),
      corner,
    );
    canvas.drawLine(
      Offset(roi.right, roi.bottom - cornerLen),
      Offset(roi.right, roi.bottom),
      corner,
    );
  }

  @override
  bool shouldRepaint(covariant _ScannerOverlayPainter oldDelegate) {
    return oldDelegate.roiNormalized != roiNormalized;
  }
}

