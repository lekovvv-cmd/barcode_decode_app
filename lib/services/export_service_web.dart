import 'dart:convert';
import 'dart:html' as html;

import 'export_service.dart';

class WebExportService implements ExportService {
  @override
  Future<void> exportCsv(String csvContent) async {
    final bytes = utf8.encode(csvContent);
    final blob = html.Blob([bytes], 'text/csv');
    final url = html.Url.createObjectUrlFromBlob(blob);
    final anchor = html.AnchorElement(href: url)
      ..download = 'scan_history.csv'
      ..click();
    html.Url.revokeObjectUrl(url);
  }
}

ExportService getExportService() => WebExportService();

