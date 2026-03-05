import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../models/tracked_barcode.dart';

class TrackingService {
  final Rect roiNormalized;
  int _nextId = 0;
  final List<TrackedBarcode> _tracks = [];

  static const int _maxFramesPerTrack = 30;
  static const int _maxFramesMissing = 10;

  TrackingService({required this.roiNormalized});

  /// Update tracks with new detections on the current frame.
  /// Returns stabilized detections that:
  /// - are inside ROI
  /// - belong to a track with framesSeen >= minFramesForDecode.
  List<Detection> trackAndFilter({
    required List<Detection> detections,
    required XFile frame,
    required double sharpness,
    int minFramesForDecode = 6,
    double sharpnessThreshold = 80,
  }) {
    final Map<int, TrackedBarcode> matchedTracks = {};
    final Set<TrackedBarcode> updatedTracks = {};

    for (var i = 0; i < detections.length; i++) {
      final det = detections[i];
      final detRect = _rectFromDetection(det);

      TrackedBarcode? bestTrack;
      double bestIou = 0.0;

      for (final track in _tracks) {
        final iou = _computeIoU(track.bbox, detRect);
        if (iou > 0.5 && iou > bestIou) {
          bestIou = iou;
          bestTrack = track;
        }
      }

      if (bestTrack != null) {
        final prevBox = bestTrack.bbox;
        bestTrack.bbox = _smoothRect(bestTrack.bbox, detRect);
        bestTrack.framesSeen += 1;
        bestTrack.framesMissing = 0;
        // Update stability based on IoU with previous box (higher IoU => more stable).
        final iouWithPrev = _computeIoU(prevBox, bestTrack.bbox);
        bestTrack.stabilityScore =
            0.8 * bestTrack.stabilityScore + 0.2 * iouWithPrev;
        if (sharpness > sharpnessThreshold) {
          bestTrack.addFrame(
            frame,
            sharpness,
            maxFrames: _maxFramesPerTrack,
          );
        }
        matchedTracks[i] = bestTrack;
        updatedTracks.add(bestTrack);
      } else {
        final newTrack = TrackedBarcode(
          id: _nextId++,
          bbox: detRect,
          framesSeen: 1,
        );
        newTrack.framesMissing = 0;
        if (sharpness > sharpnessThreshold) {
          newTrack.addFrame(
            frame,
            sharpness,
            maxFrames: _maxFramesPerTrack,
          );
        }
        _tracks.add(newTrack);
        matchedTracks[i] = newTrack;
        updatedTracks.add(newTrack);
      }
    }

    // Increment framesMissing for tracks that didn't get matched and
    // drop stale tracks to prevent memory growth.
    _tracks.removeWhere((track) {
      if (!updatedTracks.contains(track)) {
        track.framesMissing += 1;
      }
      return track.framesMissing > _maxFramesMissing;
    });

    final List<Detection> result = [];

    for (var i = 0; i < detections.length; i++) {
      final track = matchedTracks[i];
      if (track == null) continue;

      final centerX = (track.bbox.left + track.bbox.right) / 2;
      final centerY = (track.bbox.top + track.bbox.bottom) / 2;

      if (!_isInsideRoi(centerX, centerY)) continue;
      if (track.framesSeen < minFramesForDecode) continue;

      final det = detections[i];
      result.add(
        Detection(
          x1: track.bbox.left,
          y1: track.bbox.top,
          x2: track.bbox.right,
          y2: track.bbox.bottom,
          confidence: det.confidence,
          decoded: det.decoded,
        ),
      );
    }

    return result;
  }

  Rect _rectFromDetection(Detection d) {
    return Rect.fromLTRB(d.x1, d.y1, d.x2, d.y2);
  }

  Rect _smoothRect(
    Rect prev,
    Rect current, {
    double alphaPrev = 0.7,
    double alphaNew = 0.3,
  }) {
    return Rect.fromLTRB(
      alphaPrev * prev.left + alphaNew * current.left,
      alphaPrev * prev.top + alphaNew * current.top,
      alphaPrev * prev.right + alphaNew * current.right,
      alphaPrev * prev.bottom + alphaNew * current.bottom,
    );
  }

  double _computeIoU(Rect a, Rect b) {
    final double interLeft = max(a.left, b.left);
    final double interTop = max(a.top, b.top);
    final double interRight = min(a.right, b.right);
    final double interBottom = min(a.bottom, b.bottom);

    final double interWidth = max(0, interRight - interLeft);
    final double interHeight = max(0, interBottom - interTop);
    final double interArea = interWidth * interHeight;

    if (interArea <= 0) return 0.0;

    final double areaA = a.width * a.height;
    final double areaB = b.width * b.height;
    final double unionArea = areaA + areaB - interArea;

    if (unionArea <= 0) return 0.0;
    return interArea / unionArea;
  }

  bool _isInsideRoi(double x, double y) {
    return x >= roiNormalized.left &&
        x <= roiNormalized.right &&
        y >= roiNormalized.top &&
        y <= roiNormalized.bottom;
  }

  List<TrackedBarcode> get tracks => List.unmodifiable(_tracks);
}

