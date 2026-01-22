#!/usr/bin/env python3
"""
Convert OpenTelemetry JSONL exports to OCSF (Open Cybersecurity Schema Framework) format.

This unified converter handles all three telemetry signals exported by the OTel collector:
- logs.jsonl -> ocsf_logs.parquet
- traces.jsonl -> ocsf_traces.parquet
- metrics.jsonl -> ocsf_metrics.parquet

Usage:
    # Convert all signals at once
    python scripts/convert_otel_to_ocsf.py

    # Convert specific signal
    python scripts/convert_otel_to_ocsf.py --signal logs
    python scripts/convert_otel_to_ocsf.py --signal traces
    python scripts/convert_otel_to_ocsf.py --signal metrics
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)


class OTelToOCSFConverter:
    """Convert OpenTelemetry JSONL exports to OCSF format."""

    # OCSF class definitions
    OCSF_CLASSES = {
        6001: "Web Resources Activity",
        6002: "Application Lifecycle",
        6003: "API Activity",
        6004: "Web Resource Access Activity",
        99: "Metric",  # Custom for metrics
    }

    SEVERITY_MAP = {
        'TRACE': 0,
        'DEBUG': 1,
        'INFO': 2,
        'WARN': 3,
        'WARNING': 3,
        'ERROR': 4,
        'FATAL': 5,
        'CRITICAL': 6,
    }

    def convert_logs(self, input_path, output_path):
        """Convert OTel logs JSONL to OCSF parquet."""
        if not os.path.exists(input_path):
            print(f"Logs file not found: {input_path}")
            return None

        ocsf_events = []
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    otel_record = json.loads(line)
                    events = self._parse_otel_log_record(otel_record)
                    ocsf_events.extend(events)
                except json.JSONDecodeError:
                    continue

        if not ocsf_events:
            print(f"No valid log events found in {input_path}")
            return None

        return self._save_to_parquet(ocsf_events, output_path, "logs")

    def _parse_otel_log_record(self, otel_record):
        """Parse OTel log export format and extract OCSF events."""
        events = []

        # OTel log export has resourceLogs -> scopeLogs -> logRecords
        for resource_log in otel_record.get('resourceLogs', []):
            resource = resource_log.get('resource', {})
            resource_attrs = self._extract_attributes(resource.get('attributes', []))

            for scope_log in resource_log.get('scopeLogs', []):
                for log_record in scope_log.get('logRecords', []):
                    event = self._log_record_to_ocsf(log_record, resource_attrs)
                    if event:
                        events.append(event)

        return events

    def _log_record_to_ocsf(self, log_record, resource_attrs):
        """Convert a single OTel log record to OCSF format."""
        try:
            # Extract timestamp (nanoseconds to milliseconds)
            time_unix_nano = log_record.get('timeUnixNano', 0)
            if isinstance(time_unix_nano, str):
                time_unix_nano = int(time_unix_nano)
            time_ms = time_unix_nano // 1_000_000

            # Get severity
            severity_number = log_record.get('severityNumber', 9)  # Default INFO
            severity_text = log_record.get('severityText', 'INFO')
            severity_id = min(severity_number // 4, 6)  # OTel severity 1-24 -> OCSF 0-6

            # Get message body
            body = log_record.get('body', {})
            message = body.get('stringValue', '') if isinstance(body, dict) else str(body)

            # Try to parse structured log from message
            ocsf_data = {}
            if message.startswith('{'):
                try:
                    ocsf_data = json.loads(message)
                except json.JSONDecodeError:
                    pass

            # Extract attributes
            attrs = self._extract_attributes(log_record.get('attributes', []))

            # Build OCSF event
            ocsf_event = {
                # Core OCSF fields
                "class_uid": ocsf_data.get('class_uid', 6003),
                "class_name": ocsf_data.get('class_name', self.OCSF_CLASSES.get(6003)),
                "category_uid": ocsf_data.get('category_uid', 6),
                "category_name": ocsf_data.get('category_name', "Application Activity"),
                "activity_id": ocsf_data.get('activity_id', 1),
                "activity_name": ocsf_data.get('activity_name', "Unknown"),
                "type_uid": ocsf_data.get('type_uid', 600301),
                "severity_id": ocsf_data.get('severity_id', severity_id),
                "time": ocsf_data.get('time', time_ms),

                # Status fields
                "status_id": ocsf_data.get('status_id', 1),
                "status": ocsf_data.get('status', 'Success'),
                "status_code": ocsf_data.get('status_code', '200'),

                # Message and service
                "message": ocsf_data.get('message', message),
                "service": ocsf_data.get('service', resource_attrs.get('service.name', 'unknown')),
                "level": ocsf_data.get('level', severity_text),

                # Duration
                "duration": ocsf_data.get('duration'),

                # Trace context
                "trace_id": log_record.get('traceId', ocsf_data.get('trace_id')),
                "span_id": log_record.get('spanId'),
            }

            # Handle nested objects from structured log
            self._flatten_nested_objects(ocsf_event, ocsf_data)

            # Add resource attributes
            ocsf_event["service_version"] = resource_attrs.get('service.version')
            ocsf_event["host_name"] = resource_attrs.get('host.name')

            # Remove None values
            ocsf_event = {k: v for k, v in ocsf_event.items() if v is not None}

            return ocsf_event

        except Exception as e:
            return None

    def _flatten_nested_objects(self, ocsf_event, ocsf_data):
        """Flatten nested OCSF objects for ML use."""
        # Metadata
        metadata = ocsf_data.get('metadata', {})
        if metadata:
            ocsf_event["metadata"] = json.dumps(metadata)
            ocsf_event["metadata_version"] = metadata.get('version')
            product = metadata.get('product', {})
            ocsf_event["metadata_product_name"] = product.get('name')
            ocsf_event["metadata_product_version"] = product.get('version')
            ocsf_event["metadata_product_vendor_name"] = product.get('vendor_name')

        # Actor
        actor = ocsf_data.get('actor', {})
        if actor:
            ocsf_event["actor"] = json.dumps(actor)
            user = actor.get('user', {})
            ocsf_event["actor_user_uid"] = user.get('uid')
            ocsf_event["actor_user_name"] = user.get('name')
            ocsf_event["actor_user_email"] = user.get('email')
            session = actor.get('session', {})
            ocsf_event["actor_session_uid"] = session.get('uid')

        # Source endpoint
        src = ocsf_data.get('src_endpoint', {})
        if src:
            ocsf_event["src_endpoint"] = json.dumps(src)
            ocsf_event["src_endpoint_ip"] = src.get('ip')
            ocsf_event["src_endpoint_port"] = src.get('port')
            ocsf_event["src_endpoint_domain"] = src.get('domain')

        # Destination endpoint
        dst = ocsf_data.get('dst_endpoint', {})
        if dst:
            ocsf_event["dst_endpoint"] = json.dumps(dst)
            ocsf_event["dst_endpoint_ip"] = dst.get('ip')
            ocsf_event["dst_endpoint_port"] = dst.get('port')
            ocsf_event["dst_endpoint_svc_name"] = dst.get('svc_name')

        # HTTP request
        http_req = ocsf_data.get('http_request', {})
        if http_req:
            ocsf_event["http_request"] = json.dumps(http_req)
            ocsf_event["http_request_method"] = http_req.get('method')
            ocsf_event["http_request_user_agent"] = http_req.get('user_agent')
            url = http_req.get('url', {})
            ocsf_event["http_request_url_path"] = url.get('path')
            ocsf_event["http_request_url_hostname"] = url.get('hostname')
            ocsf_event["http_request_url_scheme"] = url.get('scheme')

        # HTTP response
        http_resp = ocsf_data.get('http_response', {})
        if http_resp:
            ocsf_event["http_response"] = json.dumps(http_resp)
            ocsf_event["http_response_code"] = http_resp.get('code')
            ocsf_event["http_response_status"] = http_resp.get('status')
            ocsf_event["http_response_latency"] = http_resp.get('latency')

        # Device
        device = ocsf_data.get('device', {})
        if device:
            ocsf_event["device"] = json.dumps(device)
            ocsf_event["device_hostname"] = device.get('hostname')
            ocsf_event["device_type"] = device.get('type')

        # Resources
        resources = ocsf_data.get('resources', [])
        if resources:
            ocsf_event["resources"] = json.dumps(resources)
            if len(resources) > 0:
                ocsf_event["resource_type"] = resources[0].get('type')
                ocsf_event["resource_uid"] = resources[0].get('uid')

        # Anomaly
        anomaly = ocsf_data.get('anomaly', {})
        if anomaly:
            ocsf_event["anomaly"] = json.dumps(anomaly)
            ocsf_event["anomaly_type"] = anomaly.get('type')
            ocsf_event["anomaly_severity"] = anomaly.get('severity')

        # Error
        error = ocsf_data.get('error', {})
        if error:
            ocsf_event["error"] = json.dumps(error)
            ocsf_event["error_message"] = error.get('message')
            ocsf_event["error_type"] = error.get('type')

    def convert_traces(self, input_path, output_path):
        """Convert OTel traces JSONL to OCSF parquet."""
        if not os.path.exists(input_path):
            print(f"Traces file not found: {input_path}")
            return None

        ocsf_events = []
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    otel_record = json.loads(line)
                    events = self._parse_otel_trace_record(otel_record)
                    ocsf_events.extend(events)
                except json.JSONDecodeError:
                    continue

        if not ocsf_events:
            print(f"No valid trace events found in {input_path}")
            return None

        return self._save_to_parquet(ocsf_events, output_path, "traces")

    def _parse_otel_trace_record(self, otel_record):
        """Parse OTel trace export format and extract OCSF events."""
        events = []

        for resource_span in otel_record.get('resourceSpans', []):
            resource = resource_span.get('resource', {})
            resource_attrs = self._extract_attributes(resource.get('attributes', []))

            for scope_span in resource_span.get('scopeSpans', []):
                for span in scope_span.get('spans', []):
                    event = self._span_to_ocsf(span, resource_attrs)
                    if event:
                        events.append(event)

        return events

    def _span_to_ocsf(self, span, resource_attrs):
        """Convert a single OTel span to OCSF format."""
        try:
            # Extract timing (nanoseconds to milliseconds)
            start_time = int(span.get('startTimeUnixNano', '0')) // 1_000_000
            end_time = int(span.get('endTimeUnixNano', '0')) // 1_000_000
            duration_ms = end_time - start_time if end_time > start_time else 0

            # Extract attributes
            attrs = self._extract_attributes(span.get('attributes', []))

            # Determine activity type from span kind
            span_kind = span.get('kind', 1)
            activity_map = {1: 1, 2: 2, 3: 2, 4: 1, 5: 1}  # Internal, Server, Client, Producer, Consumer
            activity_id = activity_map.get(span_kind, 1)

            # Determine status
            status = span.get('status', {})
            status_code = status.get('code', 0)
            status_id = 1 if status_code == 0 else 2

            # Build OCSF event
            ocsf_event = {
                "class_uid": 6003,
                "class_name": "API Activity",
                "category_uid": 6,
                "category_name": "Application Activity",
                "activity_id": activity_id,
                "activity_name": ["Unknown", "Create", "Read", "Update", "Delete"][min(activity_id, 4)],
                "type_uid": 600300 + activity_id,
                "severity_id": 4 if status_id == 2 else 2,
                "time": start_time,

                "status_id": status_id,
                "status": "Success" if status_id == 1 else "Failure",
                "status_code": str(attrs.get('http.status_code', 200)),

                "message": span.get('name', ''),
                "service": resource_attrs.get('service.name', 'unknown'),
                "duration": duration_ms,

                "trace_id": span.get('traceId'),
                "span_id": span.get('spanId'),
                "parent_span_id": span.get('parentSpanId'),

                # HTTP attributes if present
                "http_request_method": attrs.get('http.method'),
                "http_request_url_path": attrs.get('http.target') or attrs.get('http.url'),
                "http_response_code": attrs.get('http.status_code'),
            }

            # Remove None values
            ocsf_event = {k: v for k, v in ocsf_event.items() if v is not None}

            return ocsf_event

        except Exception as e:
            return None

    def convert_metrics(self, input_path, output_path):
        """Convert OTel metrics JSONL to OCSF parquet."""
        if not os.path.exists(input_path):
            print(f"Metrics file not found: {input_path}")
            return None

        ocsf_events = []
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    otel_record = json.loads(line)
                    events = self._parse_otel_metrics_record(otel_record)
                    ocsf_events.extend(events)
                except json.JSONDecodeError:
                    continue

        if not ocsf_events:
            print(f"No valid metric events found in {input_path}")
            return None

        return self._save_to_parquet(ocsf_events, output_path, "metrics")

    def _parse_otel_metrics_record(self, otel_record):
        """Parse OTel metrics export format and extract OCSF events."""
        events = []

        for resource_metric in otel_record.get('resourceMetrics', []):
            resource = resource_metric.get('resource', {})
            resource_attrs = self._extract_attributes(resource.get('attributes', []))

            for scope_metric in resource_metric.get('scopeMetrics', []):
                for metric in scope_metric.get('metrics', []):
                    metric_events = self._metric_to_ocsf(metric, resource_attrs)
                    events.extend(metric_events)

        return events

    def _metric_to_ocsf(self, metric, resource_attrs):
        """Convert a single OTel metric to OCSF format."""
        events = []
        metric_name = metric.get('name', 'unknown')
        metric_unit = metric.get('unit', '')
        metric_description = metric.get('description', '')

        # Handle different metric types
        data_points = []
        if 'sum' in metric:
            data_points = metric['sum'].get('dataPoints', [])
            metric_type = 'counter'
        elif 'gauge' in metric:
            data_points = metric['gauge'].get('dataPoints', [])
            metric_type = 'gauge'
        elif 'histogram' in metric:
            data_points = metric['histogram'].get('dataPoints', [])
            metric_type = 'histogram'
        elif 'summary' in metric:
            data_points = metric['summary'].get('dataPoints', [])
            metric_type = 'summary'
        else:
            return events

        for dp in data_points:
            try:
                # Get timestamp
                time_unix_nano = dp.get('timeUnixNano', 0)
                if isinstance(time_unix_nano, str):
                    time_unix_nano = int(time_unix_nano)
                time_ms = time_unix_nano // 1_000_000

                # Get value
                if 'asDouble' in dp:
                    value = dp['asDouble']
                elif 'asInt' in dp:
                    value = dp['asInt']
                elif 'sum' in dp:
                    value = dp['sum']
                elif 'count' in dp:
                    value = dp['count']
                else:
                    value = 0

                # Get labels
                labels = self._extract_attributes(dp.get('attributes', []))

                ocsf_event = {
                    "class_uid": 99,  # Custom metric class
                    "class_name": "Metric",
                    "category_uid": 6,
                    "category_name": "Application Activity",
                    "activity_id": 0,
                    "activity_name": "Observe",
                    "type_uid": 9900,
                    "severity_id": 2,  # Info
                    "time": time_ms,

                    "metric_name": metric_name,
                    "metric_value": value,
                    "metric_type": metric_type,
                    "metric_unit": metric_unit,
                    "metric_description": metric_description,

                    "service": resource_attrs.get('service.name') or labels.get('job', 'unknown'),

                    # Common metric labels
                    "endpoint": labels.get('endpoint'),
                    "method": labels.get('method'),
                    "status": labels.get('status'),
                    "job": labels.get('job'),
                    "instance": labels.get('instance'),
                }

                # Add all labels as flattened fields
                for k, v in labels.items():
                    key = f"label_{k.replace('.', '_').replace('-', '_')}"
                    ocsf_event[key] = v

                # Remove None values
                ocsf_event = {k: v for k, v in ocsf_event.items() if v is not None}

                events.append(ocsf_event)

            except Exception as e:
                continue

        return events

    def _extract_attributes(self, attrs_list):
        """Extract OTel attributes from list format to dict."""
        result = {}
        for attr in attrs_list:
            key = attr.get('key', '')
            value = attr.get('value', {})
            if 'stringValue' in value:
                result[key] = value['stringValue']
            elif 'intValue' in value:
                result[key] = int(value['intValue'])
            elif 'doubleValue' in value:
                result[key] = float(value['doubleValue'])
            elif 'boolValue' in value:
                result[key] = value['boolValue']
        return result

    def _save_to_parquet(self, events, output_path, signal_type):
        """Save events to parquet file."""
        if not events:
            print(f"Warning: No {signal_type} events to save")
            return None

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(events)
        df.to_parquet(output_path, compression='snappy')
        print(f"Saved {len(df)} {signal_type} events to {output_path}")
        print(f"  Columns: {len(df.columns)}")
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenTelemetry JSONL exports to OCSF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all signals
  python scripts/convert_otel_to_ocsf.py

  # Convert specific signal
  python scripts/convert_otel_to_ocsf.py --signal logs
  python scripts/convert_otel_to_ocsf.py --signal traces
  python scripts/convert_otel_to_ocsf.py --signal metrics

  # Custom input/output directories
  python scripts/convert_otel_to_ocsf.py --input-dir ./logs/otel --output-dir ./data
        """
    )
    parser.add_argument('--signal', choices=['logs', 'traces', 'metrics', 'all'],
                        default='all', help='Which signal to convert (default: all)')
    parser.add_argument('--input-dir', default='./logs/otel',
                        help='Directory containing OTel JSONL files (default: ./logs/otel)')
    parser.add_argument('--output-dir', default='./data',
                        help='Output directory for parquet files (default: ./data)')

    args = parser.parse_args()

    converter = OTelToOCSFConverter()
    results = {}

    if args.signal in ['logs', 'all']:
        logs_input = os.path.join(args.input_dir, 'logs.jsonl')
        logs_output = os.path.join(args.output_dir, 'ocsf_logs.parquet')
        print(f"\nConverting logs: {logs_input} -> {logs_output}")
        results['logs'] = converter.convert_logs(logs_input, logs_output)

    if args.signal in ['traces', 'all']:
        traces_input = os.path.join(args.input_dir, 'traces.jsonl')
        traces_output = os.path.join(args.output_dir, 'ocsf_traces.parquet')
        print(f"\nConverting traces: {traces_input} -> {traces_output}")
        results['traces'] = converter.convert_traces(traces_input, traces_output)

    if args.signal in ['metrics', 'all']:
        metrics_input = os.path.join(args.input_dir, 'metrics.jsonl')
        metrics_output = os.path.join(args.output_dir, 'ocsf_metrics.parquet')
        print(f"\nConverting metrics: {metrics_input} -> {metrics_output}")
        results['metrics'] = converter.convert_metrics(metrics_input, metrics_output)

    # Summary
    print("\n" + "=" * 50)
    print("Conversion Summary:")
    for signal, df in results.items():
        if df is not None:
            print(f"  {signal}: {len(df)} events, {len(df.columns)} columns")
        else:
            print(f"  {signal}: No data (file may not exist yet)")


if __name__ == '__main__':
    main()
