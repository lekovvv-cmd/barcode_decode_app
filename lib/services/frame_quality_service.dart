import 'dart:typed_data';

import 'package:image/image.dart' as img;

class FrameQualityService {
  double computeSharpness(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) return 0;

    final gray = img.grayscale(image);
    final w = gray.width;
    final h = gray.height;

    const kernel = [
      0, 1, 0,
      1, -4, 1,
      0, 1, 0,
    ];

    final values = <double>[];

    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        double lap = 0;
        var ki = 0;
        for (var ky = -1; ky <= 1; ky++) {
          for (var kx = -1; kx <= 1; kx++) {
            final px = gray.getPixel(x + kx, y + ky);
            final v = img.getLuminance(px).toDouble();
            lap += v * kernel[ki++];
          }
        }
        values.add(lap);
      }
    }

    if (values.isEmpty) return 0;

    final mean = values.reduce((a, b) => a + b) / values.length;
    final meanSq =
        values.map((v) => v * v).reduce((a, b) => a + b) / values.length;

    return meanSq - mean * mean;
  }

  double computeContrastScore(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) return 0;

    final gray = img.grayscale(image);
    final w = gray.width;
    final h = gray.height;

    final values = <double>[];
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final px = gray.getPixel(x, y);
        final v = img.getLuminance(px).toDouble();
        values.add(v);
      }
    }

    if (values.isEmpty) return 0;
    final mean = values.reduce((a, b) => a + b) / values.length;
    final meanSq =
        values.map((v) => v * v).reduce((a, b) => a + b) / values.length;
    return meanSq - mean * mean;
  }

  double computeBrightnessScore(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) return 0;

    final gray = img.grayscale(image);
    final w = gray.width;
    final h = gray.height;

    double sum = 0;
    var count = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final px = gray.getPixel(x, y);
        final v = img.getLuminance(px).toDouble();
        sum += v;
        count++;
      }
    }
    if (count == 0) return 0;
    return sum / count;
  }

  double computeQualityScore(Uint8List bytes) {
    final sharp = computeSharpness(bytes);
    final contrast = computeContrastScore(bytes);
    // brightness можно использовать для доп. нормализации, пока не мешаем.
    // final brightness = computeBrightnessScore(bytes);
    return sharp * 0.7 + contrast * 0.3;
  }
}

