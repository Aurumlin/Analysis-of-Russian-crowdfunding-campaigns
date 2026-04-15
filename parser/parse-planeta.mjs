#!/usr/bin/env node

import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright";

const DEFAULT_INPUT = "/Users/zhdanovmaxim/Downloads/ссылки планета.json";
const DEFAULT_OUTPUT = path.resolve(process.cwd(), "planeta_projects.json");
const DEFAULT_BROWSER_COUNT = Math.max(2, Math.min(6, os.cpus().length));

function printHelp() {
  console.log(`Usage:
  node parse-planeta.mjs [options]

Options:
  --input, -i       Путь до JSON со ссылками (default: ${DEFAULT_INPUT})
  --output, -o      Путь до итогового JSON (default: ${DEFAULT_OUTPUT})
  --browsers, -b    Количество параллельных браузеров (default: ${DEFAULT_BROWSER_COUNT})
  --timeout         Таймаут открытия страницы в мс (default: 45000)
  --retries         Количество повторов при ошибке (default: 2)
  --save-every      Сохранять checkpoint каждые N обработанных ссылок (default: 50)
  --headless        Запуск без UI (default)
  --headed          Запуск с UI браузеров
  --help, -h        Показать справку`);
}

function parseNumber(value, fallback, min = 0) {
  const num = Number(value);
  if (!Number.isFinite(num) || num < min) {
    return fallback;
  }
  return num;
}

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    output: DEFAULT_OUTPUT,
    browsers: DEFAULT_BROWSER_COUNT,
    timeoutMs: 45000,
    retries: 2,
    saveEvery: 50,
    headless: true,
    help: false
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      args.help = true;
      continue;
    }
    if (arg === "--input" || arg === "-i") {
      args.input = argv[i + 1];
      i += 1;
      continue;
    }
    if (arg === "--output" || arg === "-o") {
      args.output = path.resolve(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === "--browsers" || arg === "-b") {
      args.browsers = parseNumber(argv[i + 1], args.browsers, 1);
      i += 1;
      continue;
    }
    if (arg === "--timeout") {
      args.timeoutMs = parseNumber(argv[i + 1], args.timeoutMs, 1000);
      i += 1;
      continue;
    }
    if (arg === "--retries") {
      args.retries = parseNumber(argv[i + 1], args.retries, 0);
      i += 1;
      continue;
    }
    if (arg === "--save-every") {
      args.saveEvery = parseNumber(argv[i + 1], args.saveEvery, 1);
      i += 1;
      continue;
    }
    if (arg === "--headless") {
      args.headless = true;
      continue;
    }
    if (arg === "--headed") {
      args.headless = false;
      continue;
    }
    throw new Error(`Неизвестный аргумент: ${arg}`);
  }

  return args;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function numberOrNull(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function normalizeUrl(value, baseUrl = null) {
  if (typeof value !== "string") {
    return null;
  }

  const cleaned = value.trim().replaceAll("&amp;", "&");
  if (!cleaned || cleaned.startsWith("javascript:") || cleaned.startsWith("data:")) {
    return null;
  }

  try {
    if (cleaned.startsWith("//")) {
      return new URL(`https:${cleaned}`).toString();
    }
    if (baseUrl) {
      return new URL(cleaned, baseUrl).toString();
    }
    return new URL(cleaned).toString();
  } catch {
    return null;
  }
}

function stripHtml(html) {
  if (typeof html !== "string") {
    return null;
  }

  const text = html
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&quot;/gi, "\"")
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&amp;/gi, "&")
    .replace(/\s+/g, " ")
    .trim();

  return text || null;
}

function extractUrlsFromText(text, baseUrl = null) {
  if (typeof text !== "string" || !text) {
    return [];
  }

  const urls = [];
  const attrPattern = /(?:src|href|poster|image|data-src)\s*=\s*["']([^"']+)["']/gi;
  const directPattern = /\bhttps?:\/\/[^\s"'<>]+/gi;
  const protoLessPattern = /(?<=["'\s(])\/\/[^\s"'<>]+/gi;
  let match;

  while ((match = attrPattern.exec(text)) !== null) {
    const normalized = normalizeUrl(match[1], baseUrl);
    if (normalized) {
      urls.push(normalized);
    }
  }

  while ((match = directPattern.exec(text)) !== null) {
    const normalized = normalizeUrl(match[0], baseUrl);
    if (normalized) {
      urls.push(normalized);
    }
  }

  while ((match = protoLessPattern.exec(text)) !== null) {
    const normalized = normalizeUrl(match[0], baseUrl);
    if (normalized) {
      urls.push(normalized);
    }
  }

  return unique(urls);
}

function isImageUrl(url) {
  if (!url) {
    return false;
  }
  return (
    /\.(?:jpg|jpeg|png|webp|gif|bmp|svg|avif)(?:$|[?#])/i.test(url) ||
    /[?&]url=https?%3A%2F%2F[^&]+?\.(?:jpg|jpeg|png|webp|gif|bmp|svg|avif)/i.test(url)
  );
}

function isVideoUrl(url) {
  if (!url) {
    return false;
  }
  return (
    /\.(?:mp4|webm|mov|m3u8)(?:$|[?#])/i.test(url) ||
    /(youtube\.com|youtu\.be|vimeo\.com|rutube\.ru|vkvideo|\/video\/)/i.test(url)
  );
}

function classifyUrls(urls) {
  const imageUrls = [];
  const videoUrls = [];
  const otherUrls = [];

  for (const url of urls) {
    if (isImageUrl(url)) {
      imageUrls.push(url);
      continue;
    }
    if (isVideoUrl(url)) {
      videoUrls.push(url);
      continue;
    }
    otherUrls.push(url);
  }

  return {
    imageUrls: unique(imageUrls),
    videoUrls: unique(videoUrls),
    otherUrls: unique(otherUrls)
  };
}

function collectUrlLikeStrings(value, baseUrl = null, acc = [], seen = new WeakSet()) {
  if (typeof value === "string") {
    const normalized = normalizeUrl(value, baseUrl);
    if (normalized) {
      acc.push(normalized);
    }
    return acc;
  }

  if (!value || typeof value !== "object") {
    return acc;
  }

  if (seen.has(value)) {
    return acc;
  }
  seen.add(value);

  if (Array.isArray(value)) {
    for (const item of value) {
      collectUrlLikeStrings(item, baseUrl, acc, seen);
    }
    return acc;
  }

  for (const objectValue of Object.values(value)) {
    collectUrlLikeStrings(objectValue, baseUrl, acc, seen);
  }
  return acc;
}

function findBlocksFromState(state) {
  if (!state || typeof state !== "object") {
    return {
      campaign: null,
      descriptionBlock: null,
      rewardsBlock: null,
      countersBlock: null
    };
  }

  const values = Object.values(state).filter((value) => value && typeof value === "object");
  const isPlainObject = (value) => value && typeof value === "object" && !Array.isArray(value);

  const campaign =
    values.find(
      (value) =>
        isPlainObject(value) &&
        Number.isFinite(value.campaignId) &&
        typeof value.name === "string" &&
        isPlainObject(value.collectedAmount) &&
        isPlainObject(value.targetAmount)
    ) || null;

  const descriptionBlock =
    values.find(
      (value) =>
        isPlainObject(value) &&
        typeof value.description === "string" &&
        typeof value.version === "string"
    ) || null;

  const rewardsBlock =
    values.find(
      (value) =>
        isPlainObject(value) &&
        Object.prototype.hasOwnProperty.call(value, "donate") &&
        Array.isArray(value.rewards)
    ) || null;

  const countersBlock =
    values.find(
      (value) =>
        isPlainObject(value) &&
        isPlainObject(value.total) &&
        ["newsCount", "commentsCount", "backersCount"].some((key) =>
          Object.prototype.hasOwnProperty.call(value.total, key)
        )
    ) || null;

  return {
    campaign,
    descriptionBlock,
    rewardsBlock,
    countersBlock
  };
}

function findInitialStateJson(html) {
  const marker = "window.INITIAL_STATE =";
  const markerIndex = html.indexOf(marker);
  if (markerIndex < 0) {
    return null;
  }

  const objectStart = html.indexOf("{", markerIndex + marker.length);
  if (objectStart < 0) {
    return null;
  }

  let depth = 0;
  let inString = false;
  let quote = "";
  let escaped = false;

  for (let i = objectStart; i < html.length; i += 1) {
    const char = html[i];

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\") {
        escaped = true;
        continue;
      }
      if (char === quote) {
        inString = false;
      }
      continue;
    }

    if (char === "\"" || char === "'") {
      inString = true;
      quote = char;
      continue;
    }
    if (char === "{") {
      depth += 1;
      continue;
    }
    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return html.slice(objectStart, i + 1);
      }
    }
  }
  return null;
}

function tryExtractStateFromHtml(html) {
  const stateJson = findInitialStateJson(html);
  if (!stateJson) {
    return null;
  }
  try {
    return JSON.parse(stateJson);
  } catch {
    return null;
  }
}

function parseTabCounts(tabCountsRaw) {
  const tabCounts = tabCountsRaw || {};
  return {
    news: numberOrNull(tabCounts.news),
    comments: numberOrNull(tabCounts.comments),
    participants: numberOrNull(tabCounts.participants)
  };
}

function mapRewards(rewardsBlock, baseUrl) {
  if (!rewardsBlock || !Array.isArray(rewardsBlock.rewards)) {
    return [];
  }

  return rewardsBlock.rewards.map((reward) => ({
    id: reward?.id ?? null,
    fid: reward?.fid ?? null,
    name: reward?.name ?? null,
    description: reward?.description ?? null,
    descriptionHtml: reward?.descriptionHtml ?? null,
    imageUrl: normalizeUrl(reward?.imageUrl, baseUrl),
    priceForOne: reward?.priceForOne ?? null,
    purchaseCount: numberOrNull(reward?.purchaseCount),
    amount: numberOrNull(reward?.amount),
    remaining: numberOrNull(reward?.remaining),
    available: reward?.available ?? null,
    deliveryInfo: reward?.deliveryInfo ?? null,
    pickupInfo: reward?.pickupInfo ?? null,
    pickupAddress: reward?.pickupAddress ?? null,
    estimatedDeliveryDate: reward?.estimatedDeliveryDate ?? null,
    deliveryAvailable: reward?.deliveryAvailable ?? null,
    pickupAvailable: reward?.pickupAvailable ?? null,
    rewardInstruction: reward?.rewardInstruction ?? null
  }));
}

function mapDonate(donate, baseUrl) {
  if (!donate || typeof donate !== "object") {
    return null;
  }
  return {
    id: donate.id ?? null,
    fid: donate.fid ?? null,
    name: donate.name ?? null,
    descriptionHtml: donate.descriptionHtml ?? null,
    purchaseCount: numberOrNull(donate.purchaseCount),
    imageUrl: normalizeUrl(donate.imageUrl, baseUrl)
  };
}

function buildProjectRecord(taskUrl, extracted, nowIso) {
  const finalUrl = extracted.finalUrl || taskUrl;
  const campaign = extracted.campaign || null;
  const descriptionBlock = extracted.descriptionBlock || null;
  const rewardsBlock = extracted.rewardsBlock || null;
  const countersBlock = extracted.countersBlock || null;
  const tabCounts = parseTabCounts(extracted.tabCounts);
  const totals = countersBlock?.total || {};

  const descriptionHtml =
    typeof descriptionBlock?.description === "string" ? descriptionBlock.description : null;
  const descriptionText = stripHtml(descriptionHtml);
  const descriptionUrls = extractUrlsFromText(descriptionHtml || "", finalUrl);
  const descriptionClassified = classifyUrls(descriptionUrls);

  const rewards = mapRewards(rewardsBlock, finalUrl);
  const donate = mapDonate(rewardsBlock?.donate, finalUrl);

  const campaignImageUrl = normalizeUrl(campaign?.imageUrl, finalUrl);
  const campaignViewImageUrl = normalizeUrl(campaign?.viewImageUrl, finalUrl);
  const campaignVideoUrl = normalizeUrl(campaign?.videoUrl, finalUrl);

  const rewardImageUrls = rewards.map((reward) => reward.imageUrl).filter(Boolean);
  const rewardHtmlUrls = unique(
    rewards.flatMap((reward) => extractUrlsFromText(reward.descriptionHtml || "", finalUrl))
  );
  const rewardHtmlClassified = classifyUrls(rewardHtmlUrls);

  const campaignLinks = unique(
    Object.values(campaign?.links || {})
      .map((value) => normalizeUrl(value, finalUrl))
      .filter(Boolean)
  );

  const domImageUrls = unique((extracted.domMedia?.images || []).map((url) => normalizeUrl(url, finalUrl)));
  const domVideoUrls = unique((extracted.domMedia?.videos || []).map((url) => normalizeUrl(url, finalUrl)));

  const recursiveUrls = unique(collectUrlLikeStrings({ campaign, rewardsBlock, descriptionBlock }, finalUrl));
  const recursiveClassified = classifyUrls(recursiveUrls);

  const imageUrls = unique([
    campaignImageUrl,
    campaignViewImageUrl,
    extracted.meta?.ogImage ? normalizeUrl(extracted.meta.ogImage, finalUrl) : null,
    ...rewardImageUrls,
    ...descriptionClassified.imageUrls,
    ...rewardHtmlClassified.imageUrls,
    ...recursiveClassified.imageUrls,
    ...domImageUrls
  ]);

  const videoUrls = unique([
    campaignVideoUrl,
    ...descriptionClassified.videoUrls,
    ...rewardHtmlClassified.videoUrls,
    ...recursiveClassified.videoUrls,
    ...domVideoUrls
  ]);

  const externalLinks = unique([
    ...campaignLinks,
    ...descriptionClassified.otherUrls,
    ...rewardHtmlClassified.otherUrls,
    ...recursiveClassified.otherUrls
  ]).filter((url) => !imageUrls.includes(url) && !videoUrls.includes(url));

  const card = {
    campaignId: campaign?.campaignId ?? null,
    fid: campaign?.fid ?? null,
    alias: campaign?.alias ?? null,
    title: campaign?.name ?? null,
    subtitle: campaign?.shortDescription ?? null,
    status: campaign?.status ?? null,
    targetStatus: campaign?.targetStatus ?? null,
    progressInPercent: numberOrNull(campaign?.progressInPercent),
    collectedAmount: campaign?.collectedAmount ?? null,
    targetAmount: campaign?.targetAmount ?? null,
    purchaseCount: numberOrNull(campaign?.purchaseCount),
    daysToFinish: numberOrNull(campaign?.daysToFinish),
    startAt: campaign?.startAt ?? null,
    finishAt: campaign?.finishAt ?? null,
    region: campaign?.region ?? null,
    mainCategory: campaign?.mainCategory ?? null,
    ageLimit: numberOrNull(campaign?.ageLimit),
    imageUrl: campaignImageUrl,
    viewImageUrl: campaignViewImageUrl,
    videoUrl: campaignVideoUrl,
    author: campaign?.author ?? null,
    links: campaign?.links ?? null,
    additionalInfo: campaign?.additionalInfo ?? null
  };

  const counts = {
    newsCount: numberOrNull(totals.newsCount) ?? tabCounts.news,
    commentsCount: numberOrNull(totals.commentsCount) ?? tabCounts.comments,
    participantsCount: numberOrNull(totals.backersCount) ?? tabCounts.participants,
    purchasesCount: numberOrNull(campaign?.purchaseCount)
  };

  return {
    status: "ok",
    sourceUrl: taskUrl,
    finalUrl,
    scrapedAt: nowIso,
    pageTitle: extracted.pageTitle || null,
    canonicalUrl: extracted.canonicalUrl || null,
    meta: extracted.meta || null,
    card,
    description: {
      version: descriptionBlock?.version ?? null,
      html: descriptionHtml,
      text: descriptionText
    },
    counts,
    rewards: {
      donate,
      totalRewards: rewards.length,
      items: rewards
    },
    media: {
      imageUrls,
      videoUrls,
      externalLinks
    },
    raw: {
      campaign,
      descriptionBlock,
      rewardsBlock,
      countersBlock
    }
  };
}

async function scrapeCampaignPage(page, url, timeoutMs) {
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: timeoutMs });
  try {
    await page.waitForFunction(() => Boolean(window.INITIAL_STATE), { timeout: Math.min(15000, timeoutMs) });
  } catch {
    // Некоторые страницы уже отрисованы, даже если INITIAL_STATE не дождались в этом таймауте.
  }

  let extracted = await page.evaluate(() => {
    const state = window.INITIAL_STATE && typeof window.INITIAL_STATE === "object" ? window.INITIAL_STATE : null;
    const values = state ? Object.values(state).filter((value) => value && typeof value === "object") : [];
    const isPlainObject = (value) => value && typeof value === "object" && !Array.isArray(value);

    const campaign =
      values.find(
        (value) =>
          isPlainObject(value) &&
          Number.isFinite(value.campaignId) &&
          typeof value.name === "string" &&
          isPlainObject(value.collectedAmount) &&
          isPlainObject(value.targetAmount)
      ) || null;

    const descriptionBlock =
      values.find(
        (value) =>
          isPlainObject(value) &&
          typeof value.description === "string" &&
          typeof value.version === "string"
      ) || null;

    const rewardsBlock =
      values.find(
        (value) =>
          isPlainObject(value) &&
          Object.prototype.hasOwnProperty.call(value, "donate") &&
          Array.isArray(value.rewards)
      ) || null;

    const countersBlock =
      values.find(
        (value) =>
          isPlainObject(value) &&
          isPlainObject(value.total) &&
          ["newsCount", "commentsCount", "backersCount"].some((key) =>
            Object.prototype.hasOwnProperty.call(value.total, key)
          )
      ) || null;

    const unique = (array) => [...new Set(array.filter(Boolean))];

    const normalize = (value) => {
      if (typeof value !== "string") {
        return null;
      }
      const trimmed = value.trim();
      if (!trimmed || trimmed.startsWith("javascript:") || trimmed.startsWith("data:")) {
        return null;
      }
      try {
        if (trimmed.startsWith("//")) {
          return new URL(`https:${trimmed}`, window.location.href).toString();
        }
        return new URL(trimmed, window.location.href).toString();
      } catch {
        return null;
      }
    };

    const domImageUrls = unique(
      Array.from(document.querySelectorAll("img"))
        .map((image) => image.currentSrc || image.src || image.getAttribute("src"))
        .map((url) => normalize(url))
    );

    const domVideoUrls = unique([
      ...Array.from(document.querySelectorAll("video"))
        .map((video) => normalize(video.currentSrc || video.src || video.getAttribute("src")))
        .filter(Boolean),
      ...Array.from(document.querySelectorAll("video source"))
        .map((source) => normalize(source.src || source.getAttribute("src")))
        .filter(Boolean),
      ...Array.from(document.querySelectorAll("iframe"))
        .map((frame) => normalize(frame.src || frame.getAttribute("src")))
        .filter(Boolean)
        .filter((url) => /(youtube\.com|youtu\.be|vimeo\.com|rutube\.ru|video)/i.test(url))
    ]);

    const tabCounts = {};
    const textNodes = Array.from(document.querySelectorAll("a, button, span, div"))
      .slice(0, 1200)
      .map((node) => (node.textContent || "").trim())
      .filter((text) => text && text.length < 90);

    for (const text of textNodes) {
      if (!Number.isFinite(tabCounts.news)) {
        const match = text.match(/новост[ьи][^\d]{0,10}(\d+)/i);
        if (match) {
          tabCounts.news = Number(match[1]);
        }
      }
      if (!Number.isFinite(tabCounts.comments)) {
        const match = text.match(/комментар[^\d]{0,10}(\d+)/i);
        if (match) {
          tabCounts.comments = Number(match[1]);
        }
      }
      if (!Number.isFinite(tabCounts.participants)) {
        const match = text.match(/участник[^\d]{0,10}(\d+)/i);
        if (match) {
          tabCounts.participants = Number(match[1]);
        }
      }
      if (
        Number.isFinite(tabCounts.news) &&
        Number.isFinite(tabCounts.comments) &&
        Number.isFinite(tabCounts.participants)
      ) {
        break;
      }
    }

    const readMeta = (query) => {
      const element = document.querySelector(query);
      return element ? element.getAttribute("content") : null;
    };

    return {
      finalUrl: window.location.href,
      pageTitle: document.title || null,
      canonicalUrl: document.querySelector('link[rel="canonical"]')?.href || null,
      meta: {
        description: readMeta('meta[name="description"]'),
        ogTitle: readMeta('meta[property="og:title"]'),
        ogDescription: readMeta('meta[property="og:description"]'),
        ogImage: readMeta('meta[property="og:image"]')
      },
      campaign,
      descriptionBlock,
      rewardsBlock,
      countersBlock,
      tabCounts,
      domMedia: {
        images: domImageUrls,
        videos: domVideoUrls
      }
    };
  });

  if (!extracted.campaign || !extracted.descriptionBlock || !extracted.rewardsBlock) {
    const html = await page.content();
    const fallbackState = tryExtractStateFromHtml(html);
    if (fallbackState) {
      const fallbackBlocks = findBlocksFromState(fallbackState);
      extracted = {
        ...extracted,
        campaign: extracted.campaign || fallbackBlocks.campaign,
        descriptionBlock: extracted.descriptionBlock || fallbackBlocks.descriptionBlock,
        rewardsBlock: extracted.rewardsBlock || fallbackBlocks.rewardsBlock,
        countersBlock: extracted.countersBlock || fallbackBlocks.countersBlock
      };
    }
  }

  if (!extracted.campaign) {
    throw new Error("Кампания не найдена в INITIAL_STATE");
  }

  return extracted;
}

function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms <= 0) {
    return "0s";
  }
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

async function readInputUrls(inputPath) {
  if (!existsSync(inputPath)) {
    throw new Error(`Файл не найден: ${inputPath}`);
  }

  const raw = await fs.readFile(inputPath, "utf8");
  const parsed = JSON.parse(raw);
  const candidates = Array.isArray(parsed) ? parsed : parsed?.urls;

  if (!Array.isArray(candidates)) {
    throw new Error("Ожидался JSON-массив ссылок или объект с полем urls.");
  }

  const urls = unique(
    candidates
      .map((item) => (typeof item === "string" ? item.trim() : ""))
      .map((value) => normalizeUrl(value))
      .filter(Boolean)
  );

  if (urls.length === 0) {
    throw new Error("Во входном JSON не найдено валидных URL.");
  }
  return urls;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }

  const inputPath = path.resolve(args.input);
  const outputPath = path.resolve(args.output);
  const checkpointPath = `${outputPath}.checkpoint.json`;
  const urls = await readInputUrls(inputPath);
  const workerCount = Math.max(1, Math.min(args.browsers, urls.length));
  const logStep = Math.max(10, Math.floor(urls.length / 200));
  const startedAt = Date.now();

  console.log(`Ссылок: ${urls.length}`);
  console.log(`Браузеров: ${workerCount}`);
  console.log(`Выходной файл: ${outputPath}`);

  const results = new Array(urls.length);
  let cursor = 0;
  let completed = 0;
  let success = 0;
  let failed = 0;
  let lastCheckpointCompleted = 0;
  let checkpointChain = Promise.resolve();

  const getNextTask = () => {
    if (cursor >= urls.length) {
      return null;
    }
    const index = cursor;
    cursor += 1;
    return { index, url: urls[index] };
  };

  const buildSnapshot = (final = false) => ({
    generatedAt: new Date().toISOString(),
    final,
    input: inputPath,
    output: outputPath,
    config: {
      browsers: workerCount,
      timeoutMs: args.timeoutMs,
      retries: args.retries,
      headless: args.headless,
      saveEvery: args.saveEvery
    },
    stats: {
      total: urls.length,
      completed,
      success,
      failed,
      elapsedMs: Date.now() - startedAt
    },
    items: results.filter((item) => item !== undefined)
  });

  const queueCheckpointWrite = (force = false) => {
    if (!force && completed - lastCheckpointCompleted < args.saveEvery) {
      return;
    }
    lastCheckpointCompleted = completed;
    checkpointChain = checkpointChain
      .then(() => fs.writeFile(checkpointPath, JSON.stringify(buildSnapshot(false), null, 2), "utf8"))
      .catch((error) => {
        console.error(`Не удалось сохранить checkpoint: ${error.message}`);
      });
  };

  const worker = async (workerId) => {
    const browser = await chromium.launch({
      headless: args.headless
    });
    const context = await browser.newContext({
      locale: "ru-RU",
      timezoneId: "Europe/Moscow",
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    });

    await context.route("**/*", async (route) => {
      const type = route.request().resourceType();
      if (type === "image" || type === "media" || type === "font") {
        await route.abort();
        return;
      }
      await route.continue();
    });

    const page = await context.newPage();

    while (true) {
      const task = getNextTask();
      if (!task) {
        break;
      }

      let extracted = null;
      let errorMessage = null;

      for (let attempt = 1; attempt <= args.retries + 1; attempt += 1) {
        try {
          extracted = await scrapeCampaignPage(page, task.url, args.timeoutMs);
          break;
        } catch (error) {
          errorMessage = error instanceof Error ? error.message : String(error);
          if (attempt <= args.retries) {
            await page.waitForTimeout(700 * attempt);
          }
        }
      }

      const nowIso = new Date().toISOString();
      if (extracted) {
        results[task.index] = buildProjectRecord(task.url, extracted, nowIso);
        success += 1;
      } else {
        results[task.index] = {
          status: "error",
          sourceUrl: task.url,
          finalUrl: null,
          scrapedAt: nowIso,
          error: errorMessage || "Unknown error"
        };
        failed += 1;
      }

      completed += 1;
      if (completed % logStep === 0 || completed === urls.length) {
        const elapsed = Date.now() - startedAt;
        const speed = completed > 0 ? completed / (elapsed / 1000) : 0;
        const remaining = urls.length - completed;
        const etaMs = speed > 0 ? (remaining / speed) * 1000 : 0;
        console.log(
          `[${completed}/${urls.length}] ok=${success} fail=${failed} speed=${speed.toFixed(2)}/s eta=${formatDuration(
            etaMs
          )}`
        );
      }
      if (failed > 0 && completed % Math.max(25, logStep) === 0) {
        const lastErrorItem = results[task.index];
        if (lastErrorItem.status === "error") {
          console.error(`[worker ${workerId}] ${task.url} -> ${lastErrorItem.error}`);
        }
      }
      queueCheckpointWrite(false);
    }

    await page.close();
    await context.close();
    await browser.close();
  };

  try {
    await Promise.all(Array.from({ length: workerCount }, (_, index) => worker(index + 1)));
    queueCheckpointWrite(true);
    await checkpointChain;
  } catch (error) {
    queueCheckpointWrite(true);
    await checkpointChain;
    throw error;
  }

  await fs.writeFile(outputPath, JSON.stringify(buildSnapshot(true), null, 2), "utf8");
  if (existsSync(checkpointPath)) {
    await fs.unlink(checkpointPath).catch(() => {});
  }

  const finishedAt = Date.now();
  console.log(`Готово за ${formatDuration(finishedAt - startedAt)}.`);
  console.log(`Успешно: ${success}, ошибок: ${failed}.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exitCode = 1;
});
