import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

import '../models/detection.dart';
import '../models/scan_entry.dart';
import '../services/api_service.dart';
import '../services/camera_service.dart';
import '../services/export_service.dart';
import '../services/frame_quality_service.dart';
import '../services/tracking_service.dart';
import '../widgets/bounding_box_painter.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final CameraService _cameraService = CameraService();
  late final ApiService _apiService;
  late final TrackingService _trackingService;
  final FrameQualityService _frameQualityService = FrameQualityService();
  final ExportService _exportService = createExportService();

  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isProcessing = false;
  bool _isDecoding = false;
  bool _showSuccessFlash = false;
  bool _detectorFallbackEnabled = false;
  bool _isHistoryCollapsed = true;
  double _currentZoom = 1.0;

  int _consecutiveDecodeFailures = 0;

  Timer? _scanTimer;
  Timer? _flashTimer;
  Timer? _bannerTimer;
  DateTime? _lastDecodeRequestAt;
  DateTime? _lastSuccessfulDecodeAt;

  Duration _currentInterval = const Duration(milliseconds: 600);

  List<Detection> _detections = [];
  String? _decodedBannerText;
  String? _errorMessage;
  final List<ScanEntry> _scanHistory = [];
  final Set<String> _scannedBarcodes = {};
  final Map<String, DateTime> _excludedDedupIds = {};
  final List<XFile> _frameBuffer = [];

  Uint8List? _previousMotionRoi;
  double _lastMotionScore = 100.0;

  static const Rect _roiNormalized = Rect.fromLTWH(0, 0, 1, 1);
  static const Duration _decodeCooldown = Duration(milliseconds: 220);
  static const Duration _recentSuccessThrottle = Duration(milliseconds: 500);
  static const Duration _excludeTtl = Duration(seconds: 25);
  static const double _motionThreshold = 8.0;
  static const int _maxExcludeIds = 80;

  static const bool _fastDecodeEnabled = true;
  static const bool _autoZoomEnabled = false;

  @override
  void initState() {
    super.initState();
    _trackingService = TrackingService(roiNormalized: _roiNormalized);
    _apiService = ApiService(roiNormalized: _roiNormalized);
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
      final imageFile = await controller.takePicture();
      final bytes = await imageFile.readAsBytes();
      final quality = _frameQualityService.computeQualityScore(bytes);

      final motionScore = _computeMotionScore(bytes, _roiNormalized);
      _lastMotionScore = motionScore;

      _frameBuffer.add(imageFile);
      if (_frameBuffer.length > 6) {
        _frameBuffer.removeAt(0);
      }

      if (!_canSendDecodeRequest()) {
        _updateScanInterval();
        return;
      }

      if (motionScore < _motionThreshold) {
        debugPrint(
          '[SCAN] skipped motion=${motionScore.toStringAsFixed(2)} '
          'fallback=$_detectorFallbackEnabled',
        );
        _updateScanInterval();
        return;
      }

      _lastDecodeRequestAt = DateTime.now();
      _isDecoding = true;
      _pruneExcludeIds();
      final excludeIds = _currentExcludeIds();

      final latestFrame = [imageFile];
      List<Detection> detections = [];
      var usedFrames = 0;

      if (_fastDecodeEnabled) {
        detections = await _apiService.detectBarcodes(
          latestFrame,
          excludeIds: excludeIds,
        );
        usedFrames = _apiService.lastFramesSent;
      }

      if (detections.isEmpty) {
        final hasVisibleDetection = _detections.isNotEmpty;
        final multiFrames = _takeLatestFrames(
          _frameBuffer,
          maxCount: hasVisibleDetection ? 4 : 2,
        );
        detections = await _apiService.detectBarcodes(
          multiFrames,
          excludeIds: excludeIds,
        );
        usedFrames = _apiService.lastFramesSent;
      }

      _isDecoding = false;

      final stabilizedDetections = _trackingService.trackAndFilter(
        detections: detections,
        frame: imageFile,
        sharpness: quality,
        minFramesForDecode: 2,
        sharpnessThreshold: 45,
      );

      String? decoded;
      if (stabilizedDetections.isNotEmpty) {
        decoded = stabilizedDetections.first.decoded;
      }

      if (decoded != null && decoded.isNotEmpty) {
        _consecutiveDecodeFailures = 0;
        _detectorFallbackEnabled = false;
      } else {
        _consecutiveDecodeFailures += 1;
        if (_consecutiveDecodeFailures >= 5) {
          _detectorFallbackEnabled = true;
        }
      }

      debugPrint(
        '[SCAN] latency=${_apiService.lastLatencyMs}ms '
        'frames=$usedFrames '
        'jpeg=${_apiService.lastJpegBytes}B '
        'motion=${motionScore.toStringAsFixed(2)} '
        'strategy=${_apiService.lastStrategy ?? '-'} '
        'value=${decoded ?? '-'} '
        'excluded=${excludeIds.length} '
        'fallback=$_detectorFallbackEnabled',
      );

      if (_autoZoomEnabled) {
        await _applyAutoZoom(controller, stabilizedDetections);
      }

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

    final successAt = _lastSuccessfulDecodeAt;
    if (successAt != null &&
        DateTime.now().difference(successAt) < _recentSuccessThrottle) {
      return false;
    }

    final lastAt = _lastDecodeRequestAt;
    if (lastAt == null) return true;

    final elapsed = DateTime.now().difference(lastAt);
    return elapsed >= _decodeCooldown;
  }

  double _computeMotionScore(Uint8List bytes, Rect roiNorm) {
    final image = img.decodeImage(bytes);
    if (image == null) return 100.0;

    final w = image.width;
    final h = image.height;

    final x = (roiNorm.left * w).round().clamp(0, w - 1);
    final y = (roiNorm.top * h).round().clamp(0, h - 1);
    final cw = (roiNorm.width * w).round().clamp(1, w - x);
    final ch = (roiNorm.height * h).round().clamp(1, h - y);

    final cropped = img.copyCrop(image, x: x, y: y, width: cw, height: ch);
    final gray = img.grayscale(cropped);
    final small = img.copyResize(gray, width: 64, height: 64);
    final current = Uint8List(64 * 64);
    var idx = 0;
    for (var yy = 0; yy < 64; yy++) {
      for (var xx = 0; xx < 64; xx++) {
        final px = small.getPixel(xx, yy);
        current[idx++] = img.getLuminance(px).clamp(0, 255).toInt();
      }
    }

    final previous = _previousMotionRoi;
    _previousMotionRoi = current;

    if (previous == null || previous.length != current.length) {
      return 100.0;
    }

    var sum = 0;
    for (var i = 0; i < current.length; i++) {
      sum += (current[i] - previous[i]).abs();
    }

    return sum / current.length;
  }

  List<XFile> _takeLatestFrames(List<XFile> frames, {required int maxCount}) {
    if (frames.length <= maxCount) {
      return List<XFile>.from(frames);
    }
    return List<XFile>.from(frames.sublist(frames.length - maxCount));
  }

  Set<String> _currentExcludeIds() {
    return _excludedDedupIds.keys.toSet();
  }

  void _pruneExcludeIds() {
    final now = DateTime.now();
    _excludedDedupIds.removeWhere(
      (_, ts) => now.difference(ts) > _excludeTtl,
    );

    if (_excludedDedupIds.length <= _maxExcludeIds) return;
    final sorted = _excludedDedupIds.entries.toList()
      ..sort((a, b) => a.value.compareTo(b.value));
    final overflow = _excludedDedupIds.length - _maxExcludeIds;
    for (var i = 0; i < overflow; i++) {
      _excludedDedupIds.remove(sorted[i].key);
    }
  }

  @override
  void dispose() {
    _scanTimer?.cancel();
    _flashTimer?.cancel();
    _bannerTimer?.cancel();
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
      desired = const Duration(milliseconds: 160);
    } else if (hasCandidate) {
      desired = const Duration(milliseconds: 260);
    } else {
      desired = const Duration(milliseconds: 500);
    }

    if (desired == _currentInterval) return;
    _currentInterval = desired;
    _startScanTimer();
  }

  void _handleDecodedBarcode(String decoded, {String? barcodeType}) {
    final dedupId = _buildDedupId(decoded, barcodeType: barcodeType);

    if (_scannedBarcodes.contains(decoded)) {
      return;
    }
    if (_excludedDedupIds.containsKey(dedupId)) {
      return;
    }

    _scannedBarcodes.add(decoded);
    _excludedDedupIds[dedupId] = DateTime.now();
    _excludedDedupIds['raw:${decoded.trim().toLowerCase()}'] = DateTime.now();
    _pruneExcludeIds();
    _lastSuccessfulDecodeAt = DateTime.now();

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

  String _buildDedupId(String decoded, {String? barcodeType}) {
    final text = decoded.trim();
    if (text.isEmpty) return 'raw:';

    final ai01 = RegExp(r'\(01\)\s*([0-9]{14})').firstMatch(text)?.group(1);
    final ai21 = RegExp(r'\(21\)\s*([^\(\)\s]+)').firstMatch(text)?.group(1);
    if (ai01 != null && ai21 != null) {
      return 'dm:${ai01.toLowerCase()}:${ai21.toLowerCase()}';
    }
    if (ai01 != null) {
      return 'gtin:${ai01.toLowerCase()}';
    }

    final digits = text.replaceAll(RegExp(r'[^0-9]'), '');
    final type = (barcodeType ?? '').toUpperCase();
    if ((type == 'EAN13' || type == 'EAN-13') && digits.length == 13) {
      return 'gtin:0$digits';
    }
    if ((type == 'UPCA' || type == 'UPC-A') && digits.length == 12) {
      return 'gtin:0$digits';
    }
    if ((type == 'EAN8' || type == 'EAN-8') && digits.length == 8) {
      return 'ean8:$digits';
    }
    if ((type == 'UPCE' || type == 'UPC-E') &&
        (digits.length == 7 || digits.length == 8)) {
      return 'upce:$digits';
    }
    return 'raw:${text.toLowerCase()}';
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
            final isMobile = constraints.maxWidth < 760;

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
                    child: _buildHistoryPanel(isMobile: false),
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
                  height: _isHistoryCollapsed && isMobile ? 64 : 260,
                  child: _buildHistoryPanel(isMobile: isMobile),
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
            if (_showSuccessFlash)
              IgnorePointer(
                child: Container(color: const Color(0x3000FF88)),
              ),
            if (_detections.isNotEmpty)
              CustomPaint(
                painter: BoundingBoxPainter(_detections),
              ),
            if (_decodedBannerText != null)
              Positioned(
                top: 16,
                left: 24,
                right: 24,
                child: _buildDecodedBubble(_decodedBannerText!),
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

  Widget _buildDecodedBubble(String value) {
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0xCC0F1E17),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.greenAccent.withOpacity(0.85)),
        ),
        child: Text(
          value,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            color: Colors.greenAccent,
            fontWeight: FontWeight.w700,
          ),
        ),
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

  Widget _buildHistoryPanel({required bool isMobile}) {
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
            children: [
              const Icon(Icons.history_rounded, color: Colors.greenAccent, size: 18),
              const SizedBox(width: 8),
              const Text(
                'Scan History',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
              const Spacer(),
              if (isMobile)
                IconButton(
                  onPressed: () {
                    setState(() {
                      _isHistoryCollapsed = !_isHistoryCollapsed;
                    });
                  },
                  icon: Icon(
                    _isHistoryCollapsed
                        ? Icons.keyboard_arrow_up_rounded
                        : Icons.keyboard_arrow_down_rounded,
                    color: Colors.white70,
                  ),
                ),
            ],
          ),
          if (_isHistoryCollapsed && isMobile) const SizedBox.shrink(),
          if (!(_isHistoryCollapsed && isMobile)) ...[
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
        ],
      ),
    );
  }
}
