"use client";

/**
 * Macro Analyst AI Frontend — Light/Dark Toggle
 * Save as: MacroAnalystChat.jsx
 *
 * - Professional Light & Dark themes
 * - Toggle persists in localStorage ("macro_analyst_theme")
 * - Applies across all key surfaces
 * - Includes Upload (stub), Eragon promo, diagnostics, tools/datasets tabs
 */

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  TrendingUp,
  Database,
  BarChart3,
  Sparkles,
  Wrench,
  Code2,
  FileCode,
  Search,
  CheckCircle2,
  Upload,
  Link as LinkIcon,
  Info,
  Activity,
  Server,
  Sun,
  Moon,
  ShieldCheck,
} from "lucide-react";

// small helper for conditional classes
const cx = (...a) => a.filter(Boolean).join(" ");


const toolKey = (t) => `${t?.name ?? ""}::${t?.filename ?? ""}`;
const withVerification = (tool) => {
 
  const testsPassed =0; //NEED TO IMPLEMENT THIS
  return {
    ...tool,
    verification: {
      testsPassed,
      testsTotal: 5,
      stamp: Date.now(),
    },
  };
};

const MacroAnalystChat = () => {
  // THEME
  const [theme, setTheme] = useState("light"); // 'light' | 'dark'
  useEffect(() => {
    // load preference (localStorage -> system -> default light)
    const saved = typeof window !== "undefined" && localStorage.getItem("macro_analyst_theme");
    if (saved === "light" || saved === "dark") {
      setTheme(saved);
    } else if (typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      setTheme("dark");
    }
  }, []);
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("macro_analyst_theme", theme);
    }
  }, [theme]);
  const isDark = theme === "dark";
  const toggleTheme = () => setTheme((t) => (t === "light" ? "dark" : "light"));

  // CHAT STATE
  const [messages, setMessages] = useState([
    {
      type: "assistant",
      content:
        "Welcome. Ask about inflation, GDP, rates, trade, or employment. I’ll select a dataset, run tools, and summarize the evidence.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState("");
  const [progress, setProgress] = useState(0);

  const [discoveredDatasets, setDiscoveredDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [generatedTools, setGeneratedTools] = useState([]);

  const [datasetDir, setDatasetDir] = useState("");
  const [requestId, setRequestId] = useState("");
  const [connectionStatus, setConnectionStatus] = useState("disconnected"); // connecting | connected
  const [activeSidebarTab, setActiveSidebarTab] = useState("tools"); // tools | datasets | diagnostics | eragon

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => { scrollToBottom(); }, [messages, isProcessing]);

  const formatTime = (date) =>
    new Date(date).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });

  const formatNumber = (num) =>
    num >= 1_000_000 ? (num / 1_000_000).toFixed(1) + "m" : num >= 1000 ? (num / 1000).toFixed(1) + "k" : num;

  const openFilePicker = () => fileInputRef.current?.click();
  const onFileSelected = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setMessages((prev) => [
      ...prev,
      {
        type: "assistant",
        content: `File received: ${file.name} (${formatNumber(file.size)} bytes). File support will be enabled soon.`,
        timestamp: new Date(),
      },
    ]);
  };

  const processWithBackend = async (userMessage) => {
    setIsProcessing(true);
    setGeneratedTools([]);
    setDiscoveredDatasets([]);
    setSelectedDataset(null);
    setDatasetDir("");
    setProcessingStage("");
    setProgress(0);

    const USE_REAL_BACKEND = true;

    if (USE_REAL_BACKEND) {
      try {
        setConnectionStatus("connecting");
        const ws = new WebSocket("ws://localhost:8000/ws/analyze");
        wsRef.current = ws;

        ws.onopen = () => {
          setConnectionStatus("connected");
          ws.send(JSON.stringify({ query: userMessage }));
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          setProcessingStage(data.message || "");
          setProgress(typeof data.progress === "number" ? data.progress : 0);

          if (data.request_id) setRequestId(data.request_id);
          if (data.dataset_dir) setDatasetDir(data.dataset_dir);

          if (Array.isArray(data.datasets)) {
            setDiscoveredDatasets((prev) => {
              const map = new Map(prev.map((d) => [d.repo_id + "::" + d.summary, d]));
              data.datasets.forEach((d) => map.set(d.repo_id + "::" + d.summary, d));
              return Array.from(map.values());
            });
          }

          if (data.selected_dataset) setSelectedDataset(data.selected_dataset);

          // --- NEW: add tools with verifiability, keep latest version per (name::filename) key ---
          if (Array.isArray(data.tools)) {
            setGeneratedTools((prev) => {
              const map = new Map(prev.map((t) => [toolKey(t), t]));
              const augmented = data.tools.map(withVerification);
              augmented.forEach((t) => map.set(toolKey(t), t));
              return Array.from(map.values());
            });
          }

          if (data.response) {
            setMessages((prev) => [
              ...prev,
              {
                type: "assistant",
                content: data.response,
                timestamp: new Date(),
                dataset: data.dataset,
              },
            ]);
            setIsProcessing(false);
            setProcessingStage("");
            setProgress(0);
          }

          if (data.stage === "error") {
            setMessages((prev) => [
              ...prev,
              { type: "assistant", content: `Error: ${data.message}`, timestamp: new Date() },
            ]);
            setIsProcessing(false);
            setProcessingStage("");
            setProgress(0);
          }
        };

        ws.onerror = () => {
          setConnectionStatus("disconnected");
          setMessages((prev) => [
            ...prev,
            {
              type: "assistant",
              content: "Connection error. Ensure backend is running on port 8000.",
              timestamp: new Date(),
            },
          ]);
          setIsProcessing(false);
        };

        ws.onclose = () => setConnectionStatus("disconnected");
      } catch (error) {
        setConnectionStatus("disconnected");
        setMessages((prev) => [
          ...prev,
          { type: "assistant", content: `Connection error: ${String(error)}`, timestamp: new Date() },
        ]);
        setIsProcessing(false);
      }
    }
  };

  const handleSubmit = async () => {
    if (!input.trim() || isProcessing) return;
    const userMessage = { type: "user", content: input, timestamp: new Date() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    await processWithBackend(userMessage.content);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  // Timeline (labels)
  const timeline = [
    { pct: 5, label: "Search" },
    { pct: 25, label: "Select" },
    { pct: 40, label: "Download" },
    { pct: 65, label: "Tools" },
    { pct: 78, label: "Run Tools" },
    { pct: 95, label: "Finalize" },
  ];
  const activeIdx =
    timeline.findLastIndex((t) => progress >= t.pct) >= 0
      ? timeline.findLastIndex((t) => progress >= t.pct)
      : 0;

  // THEME CLASSES
  const appBg = isDark ? "bg-zinc-950 text-zinc-100" : "bg-white text-slate-800";
  const card = isDark ? "bg-zinc-900 border-zinc-800" : "bg-white border-slate-200";
  const softCard = isDark ? "bg-zinc-900/70 border-zinc-800" : "bg-white border-slate-200";
  const header = isDark ? "border-zinc-800 bg-zinc-900/80" : "border-slate-200 bg-white/90";
  const sidebar = isDark ? "border-zinc-800 bg-zinc-950/60" : "border-slate-200 bg-slate-50/80";
  const inputBox = isDark
    ? "bg-black border-zinc-800 text-zinc-100 placeholder-zinc-500 focus:ring-blue-500"
    : "bg-white border-slate-300 text-slate-900 placeholder-slate-400 focus:ring-blue-600";
  const sendBtn = isDark
    ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white hover:from-blue-500 hover:to-blue-400"
    : "bg-blue-600 text-white hover:bg-blue-700";
  const progressTrack = isDark ? "bg-zinc-800" : "bg-slate-200";
  const progressFill = isDark ? "bg-gradient-to-r from-blue-500 to-purple-500" : "bg-blue-600";
  const userBubble = isDark
    ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white"
    : "bg-blue-50 border-blue-200 text-slate-900";
  const botBubble = isDark
    ? "bg-zinc-900 text-zinc-100 border-zinc-800"
    : "bg-white text-slate-900 border-slate-200";
  const listMuted = isDark ? "text-zinc-400" : "text-slate-600";
  const subtleText = isDark ? "text-zinc-500" : "text-slate-500";

  // --- NEW: derived stats for Diagnostics summary ---
  const verifStats = React.useMemo(() => {
    const total = generatedTools.length;
    let five = 0, four = 0;
    for (const t of generatedTools) {
      if (t?.verification?.testsPassed === 5) five += 1;
      else if (t?.verification?.testsPassed === 4) four += 1;
    }
    return { total, five, four };
  }, [generatedTools]);

  return (
    <div className={cx("flex h-screen", appBg)}>
      {/* Main Column */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className={cx("border-b backdrop-blur", header)}>
          <div className="px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={cx("p-2 rounded-lg", isDark ? "bg-blue-600" : "bg-blue-600")}>
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">The Verifiable Analyst</h1>
                <p className={cx("text-xs", subtleText)}>
                  Evidence-based macro analysis •{" "}
                  <span
                    className={cx(
                      "inline-flex items-center gap-1",
                      connectionStatus === "connected"
                        ? isDark ? "text-emerald-400" : "text-emerald-600"
                        : connectionStatus === "connecting"
                        ? isDark ? "text-yellow-400" : "text-amber-600"
                        : isDark ? "text-zinc-500" : "text-slate-400"
                    )}
                  >
                    <Server className="w-3 h-3" />
                    {connectionStatus}
                  </span>
                  {requestId && (
                    <>
                      {" "}| req: <span className={cx(isDark ? "text-blue-400" : "text-blue-700")}>{requestId.slice(0, 8)}…</span>
                    </>
                  )}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className={cx(
                  "flex items-center gap-2 px-3 py-2 rounded-md border transition",
                  isDark
                    ? "bg-zinc-900 border-zinc-800 text-zinc-200 hover:bg-zinc-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"
                )}
                title="Toggle theme"
              >
                {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                <span className="text-sm">{isDark ? "Light" : "Dark"}</span>
              </button>

              {/* Upload */}
              <button
                onClick={openFilePicker}
                className={cx(
                  "flex items-center gap-2 px-3 py-2 rounded-md border transition",
                  isDark
                    ? "bg-zinc-900 border-zinc-800 text-zinc-200 hover:bg-zinc-800"
                    : "bg-white border-slate-200 text-slate-700 hover:bg-slate-50"
                )}
                title="Upload a local file (not used yet)"
              >
                <Upload className="w-4 h-4" />
                <span className="text-sm">Upload</span>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={onFileSelected}
                accept=".csv,.tsv,.parquet,.json,.jsonl,.xlsx,.xls,.feather,.arrow,.gz"
              />

              {/* Eragon */}
              <a
                href="https://eragon.ai/"
                target="_blank"
                rel="noreferrer"
                className={cx(
                  "flex items-center gap-2 px-3 py-2 rounded-md transition",
                  isDark ? "bg-blue-600 text-white hover:bg-blue-500" : "bg-blue-600 text-white hover:bg-blue-700"
                )}
                title="Eragon AI Track"
              >
                <LinkIcon className="w-4 h-4" />
                <span className="text-sm">Eragon Track</span>
              </a>
            </div>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {/* Timeline */}
          {(isProcessing || progress > 0) && (
            <div className="max-w-4xl">
              <div className={cx("flex items-center justify-between text-xs mb-2", subtleText)}>
                <span className="inline-flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5" />
                  {processingStage || "Working…"}
                </span>
                <span>{progress}%</span>
              </div>
              <div className={cx("relative h-2 rounded-full overflow-hidden", progressTrack)}>
                <div
                  className={cx("absolute inset-y-0 left-0 transition-all duration-300", progressFill)}
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className={cx("mt-2 grid grid-cols-6 text-[11px]", subtleText)}>
                {timeline.map((t, i) => (
                  <div key={t.label} className="flex items-center gap-2">
                    <div className={cx("w-2 h-2 rounded-full", i <= activeIdx ? "bg-blue-600" : isDark ? "bg-zinc-600" : "bg-slate-300")} />
                    <span>{t.label}</span>
                  </div>
                ))}
              </div>
              {datasetDir && (
                <div className={cx("mt-2 text-xs flex items-center gap-2", listMuted)}>
                  <Info className="w-3.5 h-3.5" />
                  <span className="truncate">
                    dataset_dir: <span className={cx(isDark ? "text-zinc-200" : "text-slate-800", "font-medium")}>{datasetDir}</span>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Messages */}
          {messages.map((message, index) => (
            <div
              key={index}
              className={cx("flex", message.type === "user" ? "justify-end" : "justify-start")}
            >
              <div className={cx("max-w-3xl", message.type === "user" ? "ml-12" : "mr-12")}>
                <div
                  className={cx(
                    "rounded-xl px-6 py-4 border",
                    message.type === "user" ? userBubble : botBubble
                  )}
                >
                  {message.type === "assistant" && (
                    <div className={cx("flex items-center gap-2 mb-2", isDark ? "text-blue-400" : "text-blue-700")}>
                      <Sparkles className="w-4 h-4" />
                      <span className="text-xs font-medium">ANALYST</span>
                    </div>
                  )}
                  <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                  {message.dataset && (
                    <div className={cx("mt-4 pt-4 border-t", isDark ? "border-zinc-800" : "border-slate-200")}>
                      <div className={cx("flex items-center gap-2 text-xs", listMuted)}>
                        <Database className="w-3.5 h-3.5" />
                        <span>Dataset: {message.dataset}</span>
                      </div>
                    </div>
                  )}
                </div>
                <div
                  className={cx(
                    "mt-2 text-xs",
                    message.type === "user" ? "text-right" : "text-left",
                    subtleText
                  )}
                >
                  {formatTime(message.timestamp)}
                </div>
              </div>
            </div>
          ))}

          {/* Discovered datasets */}
          {isProcessing && discoveredDatasets.length > 0 && (
            <div className="max-w-4xl">
              <div className={cx("flex items-center gap-2 text-xs mb-3", listMuted)}>
                <Search className="w-3.5 h-3.5" />
                <span>Discovered Datasets ({discoveredDatasets.length}/6)</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {discoveredDatasets.map((ds, idx) => {
                  const isSel = selectedDataset?.repo_id === ds.repo_id;
                  return (
                    <div
                      key={idx}
                      className={cx(
                        "p-4 rounded-lg border transition-colors",
                        isSel
                          ? isDark
                            ? "bg-blue-500/10 border-blue-500/50"
                            : "bg-blue-50 border-blue-200"
                          : isDark
                          ? "bg-zinc-900 border-zinc-800 hover:border-zinc-700"
                          : "bg-white border-slate-200 hover:border-slate-300"
                      )}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            {isSel && (
                              <CheckCircle2
                                className={cx("w-4 h-4", isDark ? "text-blue-400" : "text-blue-600")}
                              />
                            )}
                            <span className={cx("text-xs font-medium truncate", isDark ? "text-zinc-100" : "text-slate-900")}>
                              {ds.repo_id}
                            </span>
                          </div>
                          <p className={cx("text-xs mt-1 line-clamp-2", listMuted)}>{ds.summary}</p>
                        </div>
                        <div className="flex flex-col items-end gap-1 flex-shrink-0">
                          <span className={cx("text-xs font-medium", isDark ? "text-blue-300" : "text-blue-700")}>
                            {ds.similarity}%
                          </span>
                          <div className={cx("flex items-center gap-2 text-xs", listMuted)}>
                            <span>❤ {formatNumber(ds.likes)}</span>
                            <span>⬇ {formatNumber(ds.downloads)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Composer */}
        <div className={cx("border-t", isDark ? "border-zinc-800 bg-zinc-900/60" : "border-slate-200 bg-white")}>
          <div className="max-w-4xl mx-auto px-6 py-4">
            <div className="flex items-end gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a concise question (e.g., “How did inflation correlate with GDP for the G7 from 2000–2024?”)"
                rows="1"
                disabled={isProcessing}
                className={cx(
                  "flex-1 px-5 py-4 rounded-lg focus:outline-none focus:ring-2 resize-none disabled:opacity-60",
                  inputBox
                )}
                style={{ minHeight: "56px", maxHeight: "200px" }}
              />
              <button
                onClick={handleSubmit}
                disabled={!input.trim() || isProcessing}
                className={cx("px-4 py-3 rounded-lg transition disabled:opacity-50", sendBtn)}
                title="Send"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
            <div className={cx("mt-3 flex items-center gap-4 text-xs", subtleText)}>
              <div className="flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Dedalus Agents</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5" />
                <span>HuggingFace Snapshots</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className={cx("w-[22rem] border-l flex flex-col backdrop-blur", sidebar)}>
        {/* Tabs */}
        <div className={cx("px-6 pt-4 border-b", isDark ? "border-zinc-800 bg-zinc-900/60" : "border-slate-200 bg-white/80")}>
          <div className="flex gap-2">
            {[
              { key: "tools", label: "Tools", icon: Wrench },
              { key: "datasets", label: "Datasets", icon: Database },
              { key: "diagnostics", label: "Diagnostics", icon: Info },
              { key: "eragon", label: "Eragon", icon: Sparkles },
            ].map((tab) => {
              const Icon = tab.icon;
              const active = activeSidebarTab === tab.key;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveSidebarTab(tab.key)}
                  className={cx(
                    "flex items-center gap-2 px-3 py-2 rounded-md text-sm transition border",
                    active
                      ? isDark
                        ? "bg-blue-500/10 text-zinc-100 border-blue-500/40"
                        : "bg-blue-50 text-slate-900 border-blue-200"
                      : isDark
                      ? "bg-zinc-900 border-zinc-800 text-zinc-200 hover:bg-zinc-800"
                      : "bg-white border-slate-200 text-slate-700 hover:border-slate-300"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>
          <div className="h-3" />
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {/* Tools */}
          {activeSidebarTab === "tools" && (
            <>
              {generatedTools.length === 0 && !isProcessing ? (
                <div className={cx("text-center py-12", listMuted)}>
                  <Code2 className="w-12 h-12 mx-auto mb-3 opacity-40" />
                  <p className="text-sm">No tools yet</p>
                  <p className="text-xs mt-1">Tools appear during analysis</p>
                </div>
              ) : (
                generatedTools.map((tool, index) => (
                  <div
                    key={toolKey(tool) + index}
                    className={cx("rounded-lg p-4 border hover:transition-colors", card, isDark ? "hover:border-zinc-700" : "hover:border-slate-300")}
                  >
                    <div className="flex items-start gap-3">
                      <div className={cx("p-2 rounded-md", isDark ? "bg-blue-500/10" : "bg-blue-50")}>
                        <FileCode className={cx("w-4 h-4", isDark ? "text-blue-300" : "text-blue-700")} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2">
                          <h3 className={cx("text-sm font-medium truncate", isDark ? "text-zinc-100" : "text-slate-900")}>
                            {tool.name}
                          </h3>
                          <span className={cx("text-[10px] font-mono", listMuted)}>{tool.filename}</span>
                        </div>
                        <p className={cx("text-xs mt-1 line-clamp-3", listMuted)}>{tool.description}</p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </>
          )}

          {/* Datasets */}
          {activeSidebarTab === "datasets" && (
            <div className="space-y-3">
              {selectedDataset ? (
                <div className={cx("rounded-lg p-4 border", card)}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className={cx("w-4 h-4", isDark ? "text-blue-300" : "text-blue-700")} />
                        <h3 className={cx("text-sm font-medium truncate", isDark ? "text-zinc-100" : "text-slate-900")}>
                          {selectedDataset.repo_id}
                        </h3>
                      </div>
                      <p className={cx("text-xs mt-2", listMuted)}>{selectedDataset.summary}</p>
                      <div className={cx("mt-3 flex flex-wrap items-center gap-3 text-[11px]", listMuted)}>
                        <span>similarity: {selectedDataset.similarity}%</span>
                        <span>❤ {formatNumber(selectedDataset.likes)}</span>
                        <span>⬇ {formatNumber(selectedDataset.downloads)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className={cx("text-xs", listMuted)}>No dataset selected yet.</div>
              )}

              {discoveredDatasets.length > 0 && (
                <div className={cx("rounded-lg p-3 border", softCard)}>
                  <div className={cx("text-xs mb-2", listMuted)}>Other candidates</div>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {discoveredDatasets
                      .filter((d) => d.repo_id !== selectedDataset?.repo_id)
                      .map((ds, idx) => (
                        <div key={idx} className={cx("p-3 rounded-lg border", isDark ? "bg-zinc-950/40 border-zinc-800" : "bg-slate-50 border-slate-200")}>
                          <div className={cx("text-xs truncate", isDark ? "text-zinc-100" : "text-slate-900")}>{ds.repo_id}</div>
                          <div className={cx("text-[11px] line-clamp-2", listMuted)}>{ds.summary}</div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Diagnostics */}
          {activeSidebarTab === "diagnostics" && (
            <div className="space-y-3">
              <div className={cx("rounded-lg p-4 border", card)}>
                <div className="flex items-center justify-between">
                  <h3 className={cx("text-sm font-semibold", isDark ? "text-zinc-100" : "text-slate-900")}>Run Diagnostics</h3>
                </div>
                <div className={cx("mt-2 text-xs", listMuted)}>
                  <div className="flex items-center gap-2">
                    <Database className="w-3.5 h-3.5" />
                    <span className="truncate">
                      {datasetDir || "Dataset path will appear here during a run."}
                    </span>
                  </div>
                  <p className="text-[11px] leading-5 mt-2">
                    If loading fails, ensure the snapshot contains Parquet/CSV/JSON/Arrow/Feather/Excel.
                    Verify permissions and absolute path. Upload will eventually allow small CSV merge.
                  </p>
                </div>
              </div>

              {/* NEW: Verifiability summary */}
              <div className={cx("rounded-lg p-4 border", card)}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className={cx("w-4 h-4", isDark ? "text-emerald-300" : "text-emerald-700")} />
                    <h4 className={cx("text-sm font-semibold", isDark ? "text-zinc-100" : "text-slate-900")}>
                      Analysis Verifiability
                    </h4>
                  </div>
                  <span className={cx("text-xs", listMuted)}>
                    {verifStats.total} tool{verifStats.total === 1 ? "" : "s"}
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                  <div className={cx("p-3 rounded-md border", isDark ? "bg-zinc-900 border-zinc-800" : "bg-slate-50 border-slate-200")}> 
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Fully Verified</span>
                      <span className={cx(isDark ? "text-emerald-300" : "text-emerald-700")}>{verifStats.five}</span>
                    </div>
                    <div className={cx("mt-2 h-1.5 rounded", isDark ? "bg-zinc-800" : "bg-slate-200")}> 
                      <div style={{ width: `${verifStats.total ? (verifStats.five / Math.max(1, verifStats.total)) * 100 : 0}%` }} className={cx("h-1.5 rounded", isDark ? "bg-emerald-500" : "bg-emerald-600")} />
                    </div>
                  </div>
                  <div className={cx("p-3 rounded-md border", isDark ? "bg-zinc-900 border-zinc-800" : "bg-slate-50 border-slate-200")}> 
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Partially Verified</span>
                      <span className={cx(isDark ? "text-amber-300" : "text-amber-700")}>{verifStats.four}</span>
                    </div>
                    <div className={cx("mt-2 h-1.5 rounded", isDark ? "bg-zinc-800" : "bg-slate-200")}> 
                      <div style={{ width: `${verifStats.total ? (verifStats.four / Math.max(1, verifStats.total)) * 100 : 0}%` }} className={cx("h-1.5 rounded", isDark ? "bg-amber-500" : "bg-amber-600")} />
                    </div>
                  </div>
                </div>
                <p className={cx("mt-3 text-[11px]", listMuted)}>
                  Each tool/function is backed by unit tests.
                </p>
              </div>

              {/* NEW: List every tool with a per-function verification badge */}
              <div className={cx("rounded-lg p-4 border", card)}>
                <h4 className={cx("text-sm font-semibold mb-3", isDark ? "text-zinc-100" : "text-slate-900")}>Tool Test Coverage</h4>
                {generatedTools.length === 0 ? (
                  <div className={cx("text-xs", listMuted)}>No tools have been generated yet. Run an analysis to populate this list.</div>
                ) : (
                  <div className="space-y-2">
                    {generatedTools.map((tool, idx) => {
                      const pass = tool?.verification?.testsPassed ?? 0;
                      const total = tool?.verification?.testsTotal ?? 5;
                      const pct = Math.round((pass / total) * 100);
                      const good = pass === 5;
                      return (
                        <div key={toolKey(tool) + idx} className={cx("p-3 rounded-lg border", isDark ? "bg-zinc-950/40 border-zinc-800" : "bg-slate-50 border-slate-200")}> 
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2">
                                <FileCode className={cx("w-4 h-4", isDark ? "text-blue-300" : "text-blue-700")} />
                                <div className="truncate text-sm font-medium">{tool.name}</div>
                              </div>
                              <div className={cx("text-[11px] mt-1 truncate", listMuted)}>{tool.filename}</div>
                              {tool.description && (
                                <div className={cx("text-[11px] mt-1 line-clamp-2", listMuted)}>{tool.description}</div>
                              )}
                            </div>
                            <div className="flex flex-col items-end gap-1">
                              <span className={cx("text-xs font-semibold px-2 py-1 rounded-md border", good ? (isDark ? "text-emerald-300 border-emerald-700/40 bg-emerald-500/10" : "text-emerald-700 border-emerald-200 bg-emerald-50") : (isDark ? "text-amber-300 border-amber-700/40 bg-amber-500/10" : "text-amber-700 border-amber-200 bg-amber-50"))}>
                                {pass}/{total}
                              </span>
                              <div className={cx("w-28 h-1.5 rounded", isDark ? "bg-zinc-800" : "bg-slate-200")}> 
                                <div style={{ width: `${pct}%` }} className={cx("h-1.5 rounded", good ? (isDark ? "bg-emerald-500" : "bg-emerald-600") : (isDark ? "bg-amber-500" : "bg-amber-600"))} />
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className={cx("rounded-lg p-4 border", card)}>
                <h4 className={cx("text-sm font-semibold", isDark ? "text-zinc-100" : "text-slate-900")}>Data Quality Checklist</h4>
                <ul className={cx("mt-2 space-y-2 text-xs list-disc pl-5", listMuted)}>
                  <li>≥ 20–30 consistent periods; one frequency.</li>
                  <li>No duplicate timestamps; UTF-8; consistent delimiters.</li>
                  <li>Datetimes as native types; numeric fields parseable.</li>
                </ul>
              </div>
            </div>
          )}

          {/* Eragon promo */}
          {activeSidebarTab === "eragon" && (
            <div className="space-y-3">
              <div className={cx("rounded-lg p-5 border", card)}>
                <div className={cx("flex items-center gap-2", isDark ? "text-zinc-100" : "text-slate-900")}>
                  <Sparkles className={cx("w-5 h-5", isDark ? "text-blue-300" : "text-blue-700")} />
                  <h3 className="font-semibold">Why Join the Eragon AI Track</h3>
                </div>
                <ul className={cx("mt-3 space-y-2 text-sm", isDark ? "text-zinc-200" : "text-slate-700")}>
                  <li>• Learn how agent systems power real tools with Dedalus Labs.</li>
                  <li>• Get hands-on with AI + data integration.</li>
                  <li>• $3000 prize pool + chance to join Eragon AI’s engineering team.</li>
                  <li>• We’re seeking builders who imagine next-gen CRM.</li>
                </ul>
                <a
                  href="https://eragon.ai/"
                  target="_blank"
                  rel="noreferrer"
                  className={cx(
                    "inline-flex mt-4 items-center gap-2 px-3 py-2 rounded-md transition",
                    isDark ? "bg-blue-600 text-white hover:bg-blue-500" : "bg-blue-600 text-white hover:bg-blue-700"
                  )}
                >
                  <LinkIcon className="w-4 h-4" />
                  Learn more
                </a>
              </div>

              <div className={cx("rounded-lg p-4 border", card)}>
                <h4 className={cx("text-sm font-semibold", isDark ? "text-zinc-100" : "text-slate-900")}>Win the track: checklist</h4>
                <ul className={cx("mt-2 space-y-2 text-xs list-disc pl-5", listMuted)}>
                  <li>Evidence-first UI: show sources, schema, and tool previews.</li>
                  <li>Clear failure messages and retriable steps.</li>
                  <li>Shareable sessions and export to PDF/Markdown.</li>
                  <li>Lightweight “agent ops” panel: logs, timings, steps.</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MacroAnalystChat;
