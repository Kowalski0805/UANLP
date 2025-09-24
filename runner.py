# runner.py
import argparse, csv, os, statistics, time
from datetime import datetime
import requests, pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def build_spark(app_name: str, engine: str):
    b = (
        SparkSession.builder
        .appName(app_name)
    )
    if engine != "cuda":
        b = b.config("spark.rapids.sql.enabled", "false")
    return b.getOrCreate()

def register_udf(spark: SparkSession, engine: str):
    """
    Returns SQL expressions for body/title lemmatization.
    """
    if engine == "py":
        # Python UDF (pymorphy3)
        from spark_lp.text_ssdf import process_udf
        spark.udf.register("lemm", process_udf)  # whole-text UDF
        return "lemm(body)", "lemm(title)"
    elif engine == "java":
        # Java UDF (Morfologik/Jmorphy)
        spark.udf.registerJavaFunction("lemm_java", "org.example.MorfologikUDF")
        return "lemm_java(body)", "lemm_java(title)"
    elif engine == "cuda":
        # CUDA UDF (JNI → C++/CUDA)
        spark.udf.registerJavaFunction("lemm_gpu", "org.example.GPUUDF")
        return "lemm_gpu(body)", "lemm_gpu(title)"
    else:
        raise ValueError(f"Unknown engine: {engine}")

def read_df(spark: SparkSession, path: str, text_cols=("body","title")):
    df = spark.read.parquet(path)
    # Validate expected columns exist
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Expected text column '{c}' not found in dataset. Have: {df.columns}")
    return df

# def count_tokens(df, text_cols=("body","title")) -> int:
#     """
#     Token count via whitespace split. Adjust tokenizer if needed.
#     Uses Spark to avoid driver OOM.
#     """
#     tok_exprs = []
#     for c in text_cols:
#         tok_exprs.append(F.size(F.split(F.col(c), "\\s+")))
#     total_tokens = df.select(sum(tok_exprs).alias("tok")) \
#         .select(F.sum("tok").cast("long")).collect()[0][0]
#     return int(total_tokens or 0)

# def sum(exprs):
#     helper: sum a list of Column expressions
    # from functools import reduce
    # return reduce(lambda a,b: a+b, exprs)

def time_action(df, action: str, sink_path: str|None):
    """
    action ∈ {"count", "parquet"}
    """
    if action == "count":
        t0 = time.time()
        _ = df.count()
        return time.time() - t0
    elif action == "parquet":
        if sink_path is None:
            raise ValueError("sink_path must be provided for parquet timing")
        t0 = time.time()
        (df.write
         .mode("overwrite")
         .parquet(sink_path))
        return time.time() - t0
    else:
        raise ValueError(f"Unknown action: {action}")

def mean_ci95(samples):
    if len(samples) == 0:
        return (0.0, 0.0)
    m = statistics.mean(samples)
    if len(samples) == 1:
        return (m, 0.0)
    sd = statistics.pstdev(samples) if len(samples) < 2 else statistics.stdev(samples)
    # 95% CI using t≈2.776 for n=5; to be precise use scipy/stats, but this is fine for small n
    t_factor = 2.776 if len(samples) == 5 else 1.96
    ci = t_factor * (sd / (len(samples) ** 0.5)) if len(samples) > 1 else 0.0
    return (m, ci)

def write_csv_row(csv_path, row_dict, header_order):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if not exists:
            w.writeheader()
        w.writerow(row_dict)

def fetch_stages(spark):
    app_id = spark.sparkContext.applicationId
    ui = spark.sparkContext.uiWebUrl  # e.g., "http://localhost:4040"
    assert ui, "Spark UI must be enabled (spark.ui.enabled=true)."
    base = ui.rstrip("/")

    # 1) Get app (some deployments have multiple attempts)
    apps = requests.get(f"{base}/api/v1/applications").json()
    app = next(a for a in apps if a["id"] == app_id)
    attempt_id = app.get("attempts", [{}])[0].get("attemptId", None)

    # 2) Pull stages
    if attempt_id:
        stages = requests.get(f"{base}/api/v1/applications/{app_id}/{attempt_id}/stages").json()
    else:
        stages = requests.get(f"{base}/api/v1/applications/{app_id}/stages").json()

    # Normalize to a dataframe
    rows = []
    for s in stages:
        sub = s.get("submissionTime")
        comp = s.get("completionTime")
        def parse(ts):
            return datetime.fromisoformat(ts.replace("GMT","+00:00")) if ts else None
        t0, t1 = parse(sub), parse(comp)
        duration_ms = (t1 - t0).total_seconds()*1000 if (t0 and t1) else None

        # Aggregate some handy task metrics
        sum_metrics = s.get("executorSummary", {})  # sometimes empty
        rows.append({
            "stageId": s["stageId"],
            "name": s.get("name"),
            "status": s.get("status"),
            "numTasks": s.get("numTasks"),
            "submissionTime": sub,
            "completionTime": comp,
            "duration_ms": duration_ms,
            "executorRunTime_ms": s.get("executorRunTime", None),
            "inputBytes": s.get("inputBytes", None),
            "outputBytes": s.get("outputBytes", None),
            "shuffleRead_bytes": s.get("shuffleReadBytes", None),
            "shuffleWrite_bytes": s.get("shuffleWriteBytes", None),
            "details": s.get("details", None),
            "description": s.get("description", None),
        })
    df = pd.DataFrame(rows).sort_values("stageId")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["py","java","cuda"], required=True)
    ap.add_argument("--input", required=True, help="Path to Parquet dataset directory")
    ap.add_argument("--out_csv", default="results/runs.csv")
    ap.add_argument("--label", default="", help="e.g., small|large")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--mode", choices=["compute","e2e","both"], default="both",
                    help="compute=count() timing, e2e=parquet write timing")
    ap.add_argument("--sink", default="benchmark_output", help="Path for e2e parquet (overwritten)")
    args = ap.parse_args()

    spark = build_spark(f"lemm_bench_{args.engine}", args.engine)
    body_expr, title_expr = register_udf(spark, args.engine)

    df = read_df(spark, args.input, text_cols=("body","title"))
    # Don’t coalesce(1) — keep parallelism
    # Optional: cache to stabilize repeated runs if input is remote
    df = df.cache()
    _ = df.count()  # materialize cache

    # Token total (for tokens/sec)
    # total_tokens = count_tokens(df, text_cols=("body","title"))
    total_tokens = 175_583_276 if "big" in args.input else 807_607

    # Apply lemmatization (whole-text UDFs)
    out = df.selectExpr("*",
                        f"{body_expr} as body_vec",
                        f"{title_expr} as title_vec")

    # Warmup (helps JIT, GPU context, etc.)
    _ = out.limit(1000).count()

    compute_times, e2e_times = [], []

    if args.mode in ("compute","both"):
        for _ in range(args.repeat):
            compute_times.append(time_action(out, "count", None))

    if args.mode in ("e2e","both"):
        for _ in range(args.repeat):
            # Write only the lemmatized columns to minimize unrelated I/O
            e2e_df = out.select("body_vec","title_vec")
            e2e_times.append(time_action(e2e_df, "parquet", args.sink))

    # Aggregates
    comp_mean, comp_ci = mean_ci95(compute_times) if compute_times else (None, None)
    e2e_mean,  e2e_ci  = mean_ci95(e2e_times)    if e2e_times    else (None, None)

    # Throughput (tokens/s)
    comp_tps = (total_tokens / comp_mean) if comp_mean and comp_mean > 0 else None
    e2e_tps  = (total_tokens / e2e_mean)  if e2e_mean  and e2e_mean  > 0 else None

    row = {
        "ts": datetime.utcnow().isoformat(),
        "engine": args.engine,
        "label": args.label,
        "input_path": args.input,
        "repeat": args.repeat,
        "tokens": total_tokens,
        "compute_mean_s": round(comp_mean, 4) if comp_mean is not None else "",
        "compute_ci95_s": round(comp_ci, 4) if comp_ci is not None else "",
        "compute_tokens_per_s": round(comp_tps, 2) if comp_tps is not None else "",
        "e2e_mean_s": round(e2e_mean, 4) if e2e_mean is not None else "",
        "e2e_ci95_s": round(e2e_ci, 4) if e2e_ci is not None else "",
        "e2e_tokens_per_s": round(e2e_tps, 2) if e2e_tps is not None else "",
    }

    header = list(row.keys())
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    write_csv_row(args.out_csv, row, header)

    # Print a short summary for quick copy into paper notes
    print("\n=== Benchmark Summary ===")
    print(f"Engine: {args.engine} | Label: {args.label}")
    print(f"Tokens: {total_tokens:,}")
    if comp_mean is not None:
        print(f"[Compute] mean={comp_mean:.3f}s ±{comp_ci:.3f}s  |  {comp_tps:,.0f} tok/s")
    if e2e_mean is not None:
        print(f"[E2E]     mean={e2e_mean:.3f}s ±{e2e_ci:.3f}s  |  {e2e_tps:,.0f} tok/s")
    print(f"Results → {args.out_csv}\n")
    print("=== Spark Stages ===")
    stages_df = fetch_stages(spark)
    print(stages_df.to_string(index=False))
    print("=======================")


if __name__ == "__main__":
    main()
