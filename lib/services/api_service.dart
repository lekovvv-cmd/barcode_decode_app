import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

import '../models/detection.dart';

class ApiService {
  static const String _backendFromEnv =
      String.fromEnvironment('BACKEND_URL', defaultValue: '');

  final String baseUrl;
  final Rect roiNormalized;

  int lastLatencyMs = 0;
  int lastFramesSent = 0;
  int lastJpegBytes = 0;
  String? lastDecodedValue;
  String? lastDecodedType;
  String? lastStrategy;

  int _dynamicJpegQuality = 72;
  static const Duration _requestTimeout = Duration(seconds: 5);

  ApiService({
    String? baseUrl,
    Rect? roiNormalized,
  })  : baseUrl = _resolveBaseUrl(baseUrl),
        roiNormalized = roiNormalized ?? const Rect.fromLTWH(0, 0, 1, 1);

  static String _resolveBaseUrl(String? customUrl) {
    if (customUrl != null && customUrl.isNotEmpty) {
      return customUrl;
    }
    if (_backendFromEnv.isNotEmpty) {
      return _backendFromEnv;
    }

    if (kIsWeb) {
      return 'http://localhost:8000';
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return 'http://10.0.2.2:8000';
      default:
        return 'http://localhost:8000';
    }
  }

  Future<List<Detection>> detectBarcodes(
    List<XFile> frames, {
    Set<String>? excludeIds,
  }) async {
    if (frames.isEmpty) return [];

    lastFramesSent = frames.length;
    lastDecodedValue = null;
    lastDecodedType = null;
    lastStrategy = null;
    lastJpegBytes = 0;

    final stopwatch = Stopwatch()..start();

    try {
      final uri = Uri.parse('$baseUrl/decode');
      final request = http.MultipartRequest('POST', uri);
      final excludes = excludeIds ?? const <String>{};
      if (excludes.isNotEmpty) {
        request.fields['exclude_ids'] = jsonEncode(excludes.toList());
      }

      for (var i = 0; i < frames.length; i++) {
        final bytes = await frames[i].readAsBytes();
        final roiBytes = cropToTransportRoi(bytes);
        lastJpegBytes += roiBytes.length;

        request.files.add(
          http.MultipartFile.fromBytes(
            'frames',
            roiBytes,
            filename: 'frame_$i.jpg',
          ),
        );
      }

      final streamedResponse = await request.send().timeout(_requestTimeout);
      final body =
          await streamedResponse.stream.bytesToString().timeout(
                _requestTimeout,
              );

      if (streamedResponse.statusCode < 200 ||
          streamedResponse.statusCode >= 300) {
        debugPrint(
          'ApiService.detectBarcodes: HTTP ${streamedResponse.statusCode}, body=$body',
        );
        return [];
      }

      final decodedJson = jsonDecode(body);
      if (decodedJson is! Map<String, dynamic>) return [];

      lastStrategy = decodedJson['strategy']?.toString();
      lastDecodedType = decodedJson['type']?.toString();

      final decoded = decodedJson['decoded'];
      if (decoded == null || (decoded is String && decoded.isEmpty)) {
        return [];
      }

      lastDecodedValue = decoded.toString();
      final confidence = (decodedJson['confidence'] as num?)?.toDouble() ?? 0.0;
      final bbox = decodedJson['bbox'] as List<dynamic>?;
      final hasBbox = bbox != null && bbox.length == 4;

      final x1 = hasBbox ? (bbox[0] as num).toDouble() : roiNormalized.left;
      final y1 = hasBbox ? (bbox[1] as num).toDouble() : roiNormalized.top;
      final x2 = hasBbox ? (bbox[2] as num).toDouble() : roiNormalized.right;
      final y2 = hasBbox ? (bbox[3] as num).toDouble() : roiNormalized.bottom;

      return [
        Detection(
          x1: x1,
          y1: y1,
          x2: x2,
          y2: y2,
          confidence: confidence,
          decoded: decoded.toString(),
        ),
      ];
    } on TimeoutException {
      debugPrint('ApiService.detectBarcodes: request timeout');
      return [];
    } on FormatException {
      debugPrint('ApiService.detectBarcodes: invalid JSON response');
      return [];
    } catch (e) {
      debugPrint('ApiService.detectBarcodes: network/error: $e');
      return [];
    } finally {
      stopwatch.stop();
      lastLatencyMs = stopwatch.elapsedMilliseconds;
      _updateDynamicJpegQuality(lastLatencyMs);
    }
  }

  void _updateDynamicJpegQuality(int latencyMs) {
    if (latencyMs > 300) {
      _dynamicJpegQuality = 60;
      return;
    }
    if (latencyMs < 150) {
      _dynamicJpegQuality = 75;
      return;
    }
    _dynamicJpegQuality = 68;
  }

  // Send full frame, but cap long side for stable latency over Wi-Fi.
  Uint8List cropToTransportRoi(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) return bytes;

    const targetLongSide = 1024;
    final longSide = image.width >= image.height ? image.width : image.height;
    final scale = longSide > targetLongSide ? targetLongSide / longSide : 1.0;

    final resized = scale < 1.0
        ? img.copyResize(
            image,
            width: (image.width * scale).round(),
            height: (image.height * scale).round(),
            interpolation: img.Interpolation.linear,
          )
        : image;

    return Uint8List.fromList(
      img.encodeJpg(resized, quality: _dynamicJpegQuality),
    );
  }
}
