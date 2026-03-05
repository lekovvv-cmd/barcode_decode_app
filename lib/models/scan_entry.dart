class ScanEntry {
  final String barcode;
  final DateTime timestamp;
  final String? barcodeType;

  ScanEntry({
    required this.barcode,
    required this.timestamp,
    this.barcodeType,
  });
}

