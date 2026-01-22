#!/usr/bin/env python3
"""
Label a small subset of OCSF events for evaluating anomaly detection methods.

This is OPTIONAL - only needed if you want to compare method performance.
The main training (Part 4) doesn't need labels - it uses self-supervised learning.

Usage:
    python scripts/label_subset_for_evaluation.py
"""

import os
import sys
import pandas as pd


def label_evaluation_subset(ocsf_events_path, sample_size=1000):
    """
    Label a small subset for evaluating anomaly detection methods.

    Args:
        ocsf_events_path: Path to OCSF Parquet file
        sample_size: How many events to label for evaluation

    Returns:
        Small labeled DataFrame for validation
    """
    # Check if input file exists
    if not os.path.exists(ocsf_events_path):
        print(f"Input file not found: {ocsf_events_path}")
        print()
        print("Please run the OCSF conversion first:")
        print("  python scripts/convert_otel_to_ocsf.py")
        return None

    df = pd.read_parquet(ocsf_events_path)

    if len(df) == 0:
        print(f"No events found in {ocsf_events_path}")
        print("Make sure docker compose services are running and generating logs.")
        return None

    print(f"Loaded {len(df)} events with columns: {list(df.columns)}")

    # Sample a subset
    eval_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

    # Initialize anomaly labels
    eval_df['is_anomaly'] = 0
    eval_df['anomaly_reason'] = ''

    anomaly_count = 0

    # Label based on available columns

    # 1. Check for explicit anomaly field (from new structured logs)
    if 'anomaly_type' in eval_df.columns:
        mask = eval_df['anomaly_type'].notna()
        eval_df.loc[mask, 'is_anomaly'] = 1
        eval_df.loc[mask, 'anomaly_reason'] = 'explicit_anomaly_' + eval_df.loc[mask, 'anomaly_type'].astype(str)
        anomaly_count += mask.sum()
        print(f"  Found {mask.sum()} events with explicit anomaly markers")

    # 2. High latency (if duration field exists)
    if 'duration' in eval_df.columns:
        # Duration might be in ms
        duration_threshold = 2000  # 2 seconds in ms
        mask = eval_df['duration'].notna() & (eval_df['duration'] > duration_threshold)
        eval_df.loc[mask, 'is_anomaly'] = 1
        eval_df.loc[mask & (eval_df['anomaly_reason'] == ''), 'anomaly_reason'] = 'high_latency'
        new_anomalies = mask.sum() - (eval_df.loc[mask, 'anomaly_reason'].str.contains('explicit', na=False)).sum()
        if new_anomalies > 0:
            print(f"  Found {new_anomalies} events with high latency (>{duration_threshold}ms)")

    # 3. HTTP response latency (alternative duration field)
    if 'http_response_latency' in eval_df.columns:
        mask = eval_df['http_response_latency'].notna() & (eval_df['http_response_latency'] > 2000)
        eval_df.loc[mask, 'is_anomaly'] = 1
        eval_df.loc[mask & (eval_df['anomaly_reason'] == ''), 'anomaly_reason'] = 'high_response_latency'

    # 4. Error status (failures)
    if 'status_id' in eval_df.columns:
        mask = eval_df['status_id'] == 2  # Failure
        eval_df.loc[mask, 'is_anomaly'] = 1
        eval_df.loc[mask & (eval_df['anomaly_reason'] == ''), 'anomaly_reason'] = 'status_failure'
        print(f"  Found {mask.sum()} events with failure status")

    # 5. HTTP error codes (4xx, 5xx)
    if 'http_response_code' in eval_df.columns:
        mask = eval_df['http_response_code'].notna() & (eval_df['http_response_code'] >= 400)
        eval_df.loc[mask, 'is_anomaly'] = 1
        eval_df.loc[mask & (eval_df['anomaly_reason'] == ''), 'anomaly_reason'] = 'http_error'

    # 6. High severity (ERROR, CRITICAL)
    if 'severity_id' in eval_df.columns:
        mask = eval_df['severity_id'] >= 4  # ERROR or higher
        eval_df.loc[mask, 'is_anomaly'] = 1
        eval_df.loc[mask & (eval_df['anomaly_reason'] == ''), 'anomaly_reason'] = 'high_severity'
        print(f"  Found {mask.sum()} events with high severity (ERROR+)")

    # 7. Check message content for specific anomaly keywords (not generic "error")
    # Only check INFO-level messages to avoid double-counting ERROR logs
    if 'message' in eval_df.columns:
        # Use specific keywords that indicate anomalies, not generic "error"
        anomaly_keywords = ['timeout', 'leak', 'storm', 'exhausted', 'overflow', 'oom', 'killed']
        info_mask = eval_df['severity_id'] <= 2 if 'severity_id' in eval_df.columns else True
        for keyword in anomaly_keywords:
            mask = info_mask & eval_df['message'].str.lower().str.contains(keyword, na=False)
            new_anomalies = mask & (eval_df['is_anomaly'] == 0)
            eval_df.loc[mask, 'is_anomaly'] = 1
            eval_df.loc[new_anomalies & (eval_df['anomaly_reason'] == ''), 'anomaly_reason'] = f'keyword_{keyword}'

    # Summary
    total_anomalies = eval_df['is_anomaly'].sum()
    print(f"\nLabeled {total_anomalies} / {len(eval_df)} events as anomalies for evaluation")
    print(f"Anomaly rate: {total_anomalies/len(eval_df)*100:.2f}%")

    if total_anomalies > 0:
        print(f"\nAnomaly reasons:")
        reason_counts = eval_df[eval_df['is_anomaly'] == 1]['anomaly_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")

    # Save small evaluation set
    output_path = 'data/ocsf_eval_subset.parquet'
    eval_df.to_parquet(output_path)
    print(f"\nSaved evaluation subset to {output_path}")

    return eval_df


if __name__ == '__main__':
    # Generate small labeled subset for Part 6 evaluation (optional)
    result = label_evaluation_subset('data/ocsf_logs.parquet', sample_size=1000)
    if result is None:
        sys.exit(1)
