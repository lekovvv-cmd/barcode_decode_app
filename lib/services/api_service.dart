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
  String? lastDecodedValue;
  String? lastDecodedType;
  String? lastStrategy;

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

  Future<List<Detection>> detectBarcodes(List<XFile> frames) async {
    if (frames.isEmpty) return [];

    lastFramesSent = frames.length;
    lastDecodedValue = null;
    lastDecodedType = null;
    lastStrategy = null;

    final stopwatch = Stopwatch()..start();

    try {
      final uri = Uri.parse('$baseUrl/decode');
      final request = http.MultipartRequest('POST', uri);

      for (var i = 0; i < frames.length; i++) {
        final bytes = await frames[i].readAsBytes();
        final roiBytes = cropToRoi(bytes, roiNormalized);
        request.files.add(
          http.MultipartFile.fromBytes(
            'frames',
            roiBytes,
            filename: 'frame_$i.jpg',
          ),
        );
      }

      final streamedResponse =
          await request.send().timeout(const Duration(seconds: 3));
      final body =
          await streamedResponse.stream.bytesToString().timeout(
                const Duration(seconds: 3),
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

      return [
        Detection(
          x1: roiNormalized.left,
          y1: roiNormalized.top,
          x2: roiNormalized.right,
          y2: roiNormalized.bottom,
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
    }
  }

  Uint8List cropToRoi(Uint8List bytes, Rect roiNorm) {
    final image = img.decodeImage(bytes);
    if (image == null) return bytes;

    final w = image.width;
    final h = image.height;

    final x = (roiNorm.left * w).round();
    final y = (roiNorm.top * h).round();
    final cw = (roiNorm.width * w).round();
    final ch = (roiNorm.height * h).round();

    final cropped = img.copyCrop(image, x: x, y: y, width: cw, height: ch);
    return Uint8List.fromList(img.encodeJpg(cropped, quality: 90));
  }
}

