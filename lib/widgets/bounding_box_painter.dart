import 'package:flutter/material.dart';

import '../models/detection.dart';

class BoundingBoxPainter extends CustomPainter {
  final List<Detection> detections;

  BoundingBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final textPainter = TextPainter(
      textAlign: TextAlign.left,
      textDirection: TextDirection.ltr,
    );

    for (final d in detections) {
      final rect = Rect.fromLTRB(
        d.x1 * size.width,
        d.y1 * size.height,
        d.x2 * size.width,
        d.y2 * size.height,
      );

      canvas.drawRect(rect, boxPaint);

      final label = '${(d.confidence * 100).toStringAsFixed(1)}%';
      textPainter.text = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.greenAccent,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      );
      textPainter.layout();

      final offset = Offset(rect.left, rect.top - textPainter.height - 4);
      textPainter.paint(canvas, offset);
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}

