import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import 'export_service.dart';

class IoExportService implements ExportService {
  @override
  Future<void> exportCsv(String csvContent) async {
    final dir = await getTemporaryDirectory();
    final file = File(
      '${dir.path}/scan_history_${DateTime.now().millisecondsSinceEpoch}.csv',
    );
    await file.writeAsString(csvContent);
    await Share.shareXFiles([XFile(file.path)], text: 'Barcode scan history');
  }
}

ExportService getExportService() => IoExportService();

