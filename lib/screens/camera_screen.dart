import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/detection.dart';
import '../models/scan_entry.dart';
import '../models/tracked_barcode.dart';
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
  bool _torchOn = false;
  double _currentZoom = 1.0;
  Timer? _scanTimer;
  Duration _currentInterval = const Duration(milliseconds: 800);

  List<Detection> _detections = [];
  String? _lastDecodedText;
  String? _errorMessage;
  final List<ScanEntry> _scanHistory = [];
  final Set<String> _scannedBarcodes = {};
  final List<XFile> _frameBuffer = [];

  // Normalized ROI in the center of the frame (x, y, w, h all in 0..1).
  static const Rect _roiNormalized = Rect.fromLTWH(0.2, 0.3, 0.6, 0.4);

  late final AnimationController _scanLineController;

  @override
  void initState() {
    super.initState();
    _trackingService = TrackingService(roiNormalized: _roiNormalized);
    _apiService = ApiService(roiNormalized: _roiNormalized);
    _scanLineController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
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
      if (_frameBuffer.length > 6) {
        _frameBuffer.removeAt(0);
      }

      if (_frameBuffer.isEmpty) {
        return;
      }

      final detections =
          await _apiService.detectBarcodes(List<XFile>.from(_frameBuffer));

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
        if (decoded != null && decoded.isNotEmpty) {
          _handleDecodedBarcode(decoded);
        }
      });

      _updateScanInterval();
    } catch (e) {
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

  @override
  void dispose() {
    _scanTimer?.cancel();
    _scanLineController.dispose();
    _cameraController?.dispose();
    super.dispose();
  }

  void _startScanTimer() {
    _scanTimer?.cancel();
    _scanTimer =
        Timer.periodic(_currentInterval, (_) => _captureAndDetect());
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
      desired = const Duration(milliseconds: 200);
    } else if (hasCandidate) {
      desired = const Duration(milliseconds: 400);
    } else {
      desired = const Duration(milliseconds: 800);
    }

    if (desired == _currentInterval) return;
    _currentInterval = desired;
    _startScanTimer();
  }

  void _handleDecodedBarcode(String decoded) {
    if (_scannedBarcodes.contains(decoded)) {
      return;
    }

    _scannedBarcodes.add(decoded);
    final now = DateTime.now();
    _scanHistory.insert(
      0,
      ScanEntry(barcode: decoded, timestamp: now),
    );
    _lastDecodedText = decoded;
    _playFeedback();
  }

  Future<void> _exportCsv() async {
    if (_scanHistory.isEmpty) {
      setState(() {
        _errorMessage = 'Nothing to export yet.';
      });
      return;
    }

    final buffer = StringBuffer('barcode,timestamp\n');
    for (final entry in _scanHistory) {
      final ts = _formatTimestamp(entry.timestamp);
      buffer.writeln('${entry.barcode},$ts');
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

  Future<List<XFile>> _prepareFramesForBackend(
    TrackedBarcode track, {
    int maxFrames = 10,
  }) async {
    final bestFrames = track.getBestFrames(maxCount: maxFrames);
    final List<XFile> prepared = [];

    for (final frame in bestFrames) {
      final bytes = await frame.readAsBytes();
      final cropped = _apiService.cropToRoi(bytes, _roiNormalized);

      final dir = await Directory.systemTemp.createTemp('barcode_roi_');
      final file = File(
        '${dir.path}/frame_${DateTime.now().microsecondsSinceEpoch}.jpg',
      );
      await file.writeAsBytes(cropped);
      prepared.add(XFile(file.path));
    }

    return prepared;
  }

  Future<void> _toggleTorch() async {
    final controller = _cameraController;
    if (controller == null || !_isCameraInitialized) return;
    try {
      _torchOn = !_torchOn;
      await controller
          .setFlashMode(_torchOn ? FlashMode.torch : FlashMode.off);
      setState(() {});
    } catch (_) {}
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

    // Smooth zoom to avoid jitter.
    final smoothed =
        0.8 * _currentZoom + 0.2 * targetZoom;
    final clamped = smoothed.clamp(1.0, 4.0).toDouble();

    if ((clamped - _currentZoom).abs() > 0.02) {
      _currentZoom = clamped;
      await controller.setZoomLevel(_currentZoom);
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Barcode AI Scanner'),
      ),
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: _buildCameraPreview(controller),
              ),
            ),
            const SizedBox(height: 8),
            _buildDecodedText(),
            const SizedBox(height: 8),
            _buildHistoryPanel(),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraPreview(CameraController? controller) {
    if (_errorMessage != null) {
      return Text(
        _errorMessage!,
        style: const TextStyle(color: Colors.redAccent),
      );
    }

    if (!_isCameraInitialized ||
        controller == null ||
        !controller.value.isInitialized) {
      return const CircularProgressIndicator();
    }

    return AspectRatio(
      aspectRatio: controller.value.aspectRatio,
      child: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(controller),
          Positioned(
            top: 16,
            right: 16,
            child: IconButton(
              icon: Icon(
                _torchOn ? Icons.flash_on : Icons.flash_off,
                color: Colors.white,
              ),
              onPressed: _toggleTorch,
            ),
          ),
          CustomPaint(
            painter: _RoiPainter(_roiNormalized),
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
        ],
      ),
    );
  }

  Widget _buildDecodedText() {
    if (_lastDecodedText == null) {
      return const Text(
        'No barcode detected yet.',
        style: TextStyle(color: Colors.white70),
      );
    }

    return Column(
      children: [
        const Text(
          'Last decoded barcode:',
          style: TextStyle(color: Colors.white70),
        ),
        const SizedBox(height: 4),
        Text(
          _lastDecodedText!,
          style: const TextStyle(
            color: Colors.greenAccent,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _buildHistoryPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
      decoration: BoxDecoration(
        color: Colors.grey.shade900,
        border: const Border(
          top: BorderSide(color: Colors.white12),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Scan history',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
              TextButton.icon(
                onPressed: _exportCsv,
                icon: const Icon(Icons.upload_file, size: 18),
                label: const Text('Export CSV'),
              ),
            ],
          ),
          const SizedBox(height: 4),
          SizedBox(
            height: 150,
            child: _scanHistory.isEmpty
                ? const Center(
                    child: Text(
                      'No scans yet.',
                      style: TextStyle(color: Colors.white54),
                    ),
                  )
                : ListView.separated(
                    itemCount: _scanHistory.length,
                    separatorBuilder: (_, __) => const Divider(
                      height: 1,
                      color: Colors.white12,
                    ),
                    itemBuilder: (context, index) {
                      final entry = _scanHistory[index];
                      return ListTile(
                        dense: true,
                        contentPadding: EdgeInsets.zero,
                        title: Text(
                          entry.barcode,
                          style: const TextStyle(
                            color: Colors.greenAccent,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        subtitle: Text(
                          _formatTimestamp(entry.timestamp),
                          style: const TextStyle(color: Colors.white70),
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

class _RoiPainter extends CustomPainter {
  final Rect roiNormalized;

  _RoiPainter(this.roiNormalized);

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Rect.fromLTWH(
      roiNormalized.left * size.width,
      roiNormalized.top * size.height,
      roiNormalized.width * size.width,
      roiNormalized.height * size.height,
    );

    final paint = Paint()
      ..color = Colors.white.withOpacity(0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawRect(rect, paint);
  }

  @override
  bool shouldRepaint(covariant _RoiPainter oldDelegate) {
    return oldDelegate.roiNormalized != roiNormalized;
  }
}

