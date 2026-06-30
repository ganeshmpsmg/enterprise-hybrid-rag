type HealthStatus = {
  status: string;
  version: string;
  vector_store_status: string;
  total_documents: number;
  total_chunks: number;
  uptime_seconds: number;
};

async function getHealthStatus(): Promise<HealthStatus> {
  try {
    const response = await fetch("/api/v1/health", {
      cache: "no-store",
      next: { revalidate: 0 },
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    return (await response.json()) as HealthStatus;
  } catch {
    return {
      status: "unavailable",
      version: "unknown",
      vector_store_status: "unavailable",
      total_documents: 0,
      total_chunks: 0,
      uptime_seconds: 0,
    };
  }
}

export default async function Home() {
  const health = await getHealthStatus();
  const isBackendHealthy = health.status === "ok" || health.status === "degraded";
  const badgeText = isBackendHealthy
    ? health.status === "ok"
      ? "Backend online"
      : "Backend starting"
    : "Backend unavailable";

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 px-6 py-16 text-zinc-900 dark:bg-black dark:text-zinc-100">
      <div className="w-full max-w-3xl rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-zinc-500">
              Enterprise Hybrid RAG
            </p>
            <h1 className="mt-2 text-3xl font-semibold">
              Frontend and backend are connected through the deployment route.
            </h1>
          </div>
          <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-sm font-medium text-emerald-600 dark:text-emerald-400">
            {badgeText}
          </span>
        </div>

        <div className="mt-8 grid gap-4 rounded-xl border border-zinc-200 bg-zinc-50 p-5 dark:border-zinc-800 dark:bg-zinc-900/80 md:grid-cols-3">
          <div>
            <p className="text-sm text-zinc-500">Backend status</p>
            <p className="mt-1 text-lg font-semibold">{health.status}</p>
          </div>
          <div>
            <p className="text-sm text-zinc-500">Vector store</p>
            <p className="mt-1 text-lg font-semibold">{health.vector_store_status}</p>
          </div>
          <div>
            <p className="text-sm text-zinc-500">Version</p>
            <p className="mt-1 text-lg font-semibold">{health.version}</p>
          </div>
        </div>

        <p className="mt-6 text-base leading-7 text-zinc-600 dark:text-zinc-400">
          {isBackendHealthy
            ? "The deployment is responding normally. You can continue using the app without seeing a broken connection state."
            : "The backend is currently unavailable. The UI will remain informative instead of failing with a server or connection error."}
        </p>
      </div>
    </main>
  );
}
