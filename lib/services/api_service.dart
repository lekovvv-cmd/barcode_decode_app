import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../models/detection.dart';

class ApiService {
  final String baseUrl;
  final Rect roiNormalized;

  ApiService({
    this.baseUrl = 'http://localhost:8000',
    Rect? roiNormalized,
  }) : roiNormalized = roiNormalized ?? const Rect.fromLTWH(0, 0, 1, 1);

  Future<List<Detection>> detectBarcodes(List<XFile> frames) async {
    // TODO: Replace with real HTTP call to your backend.
    // For detection we typically send full frames or thumbnails.

    await Future.delayed(const Duration(milliseconds: 600));

    final mockResponse = <String, dynamic>{
      'detections': [
        {
          'bbox': [0.25, 0.35, 0.75, 0.55],
          'confidence': 0.91,
          'decoded': '5901234123457',
        },
      ],
    };

    return Detection.listFromJson(mockResponse);
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

    final cropped =
        img.copyCrop(image, x: x, y: y, width: cw, height: ch);
    return Uint8List.fromList(img.encodeJpg(cropped, quality: 90));
  }
}

