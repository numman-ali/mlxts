/**
 * Small Prometheus text-format registry used by @mlxts/serve.
 * @module
 */

export type MetricType = "counter" | "gauge" | "histogram";

export type MetricDescriptor = {
  name: string;
  help: string;
  type: MetricType;
  labelNames: readonly string[];
};

type MetricPoint = {
  labels: readonly string[];
  value: number;
};

type HistogramPoint = {
  labels: readonly string[];
  bucketCounts: number[];
  count: number;
  sum: number;
};

export function metricKey(labels: readonly string[]): string {
  return JSON.stringify(labels);
}

function escapeHelp(value: string): string {
  return value.replaceAll("\\", "\\\\").replaceAll("\n", "\\n");
}

function escapeLabelValue(value: string): string {
  return value.replaceAll("\\", "\\\\").replaceAll("\n", "\\n").replaceAll('"', '\\"');
}

function formatNumber(value: number): string {
  return value.toString();
}

function formatLabels(labelNames: readonly string[], values: readonly string[]): string {
  if (labelNames.length === 0) {
    return "";
  }
  const labels: string[] = [];
  for (let index = 0; index < labelNames.length; index += 1) {
    const name = labelNames[index];
    const value = values[index];
    if (name !== undefined && value !== undefined) {
      labels.push(`${name}="${escapeLabelValue(value)}"`);
    }
  }
  return `{${labels.join(",")}}`;
}

function formatLabelsWithExtra(
  labelNames: readonly string[],
  values: readonly string[],
  extraName: string,
  extraValue: string,
): string {
  return formatLabels([...labelNames, extraName], [...values, extraValue]);
}

function formatHeader(descriptor: MetricDescriptor): string[] {
  return [
    `# HELP ${descriptor.name} ${escapeHelp(descriptor.help)}`,
    `# TYPE ${descriptor.name} ${descriptor.type}`,
  ];
}

function sortedPoints(points: ReadonlyMap<string, MetricPoint>): MetricPoint[] {
  return [...points.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map((entry) => entry[1]);
}

function sortedHistogramPoints(points: ReadonlyMap<string, HistogramPoint>): HistogramPoint[] {
  return [...points.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map((entry) => entry[1]);
}

export class NumberMetric {
  readonly descriptor: MetricDescriptor;
  readonly #points = new Map<string, MetricPoint>();

  constructor(descriptor: MetricDescriptor) {
    this.descriptor = descriptor;
  }

  add(labels: readonly string[], value: number): void {
    const key = metricKey(labels);
    const previous = this.#points.get(key);
    if (previous === undefined) {
      this.#points.set(key, { labels: [...labels], value });
      return;
    }
    this.#points.set(key, { labels: previous.labels, value: previous.value + value });
  }

  set(labels: readonly string[], value: number): void {
    this.#points.set(metricKey(labels), { labels: [...labels], value });
  }

  format(): string[] {
    const lines = formatHeader(this.descriptor);
    for (const point of sortedPoints(this.#points)) {
      lines.push(
        `${this.descriptor.name}${formatLabels(
          this.descriptor.labelNames,
          point.labels,
        )} ${formatNumber(point.value)}`,
      );
    }
    return lines;
  }
}

export class HistogramMetric {
  readonly descriptor: MetricDescriptor;
  readonly #points = new Map<string, HistogramPoint>();
  readonly #buckets: readonly number[];

  constructor(descriptor: MetricDescriptor, buckets: readonly number[]) {
    this.descriptor = descriptor;
    this.#buckets = buckets;
  }

  observe(labels: readonly string[], value: number): void {
    const key = metricKey(labels);
    let point = this.#points.get(key);
    if (point === undefined) {
      point = {
        labels: [...labels],
        bucketCounts: Array.from({ length: this.#buckets.length }, () => 0),
        count: 0,
        sum: 0,
      };
      this.#points.set(key, point);
    }
    point.count += 1;
    point.sum += value;
    for (let index = 0; index < this.#buckets.length; index += 1) {
      const bucket = this.#buckets[index];
      if (bucket !== undefined && value <= bucket) {
        point.bucketCounts[index] = (point.bucketCounts[index] ?? 0) + 1;
      }
    }
  }

  format(): string[] {
    const lines = formatHeader(this.descriptor);
    for (const point of sortedHistogramPoints(this.#points)) {
      for (let index = 0; index < this.#buckets.length; index += 1) {
        const bucket = this.#buckets[index];
        const count = point.bucketCounts[index];
        if (bucket !== undefined && count !== undefined) {
          lines.push(
            `${this.descriptor.name}_bucket${formatLabelsWithExtra(
              this.descriptor.labelNames,
              point.labels,
              "le",
              bucket.toString(),
            )} ${formatNumber(count)}`,
          );
        }
      }
      lines.push(
        `${this.descriptor.name}_bucket${formatLabelsWithExtra(
          this.descriptor.labelNames,
          point.labels,
          "le",
          "+Inf",
        )} ${formatNumber(point.count)}`,
      );
      lines.push(
        `${this.descriptor.name}_sum${formatLabels(
          this.descriptor.labelNames,
          point.labels,
        )} ${formatNumber(point.sum)}`,
      );
      lines.push(
        `${this.descriptor.name}_count${formatLabels(
          this.descriptor.labelNames,
          point.labels,
        )} ${formatNumber(point.count)}`,
      );
    }
    return lines;
  }
}
