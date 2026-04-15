#!/usr/bin/env node

import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright";

const DISCOVER_SEED_URL = "https://boomstarter.ru/discover?v2=true&scope=ended";
const DEFAULT_LINKS_OUTPUT = path.resolve(process.cwd(), "boomstarter_links.json");
const DEFAULT_PROJECTS_OUTPUT = path.resolve(process.cwd(), "boomstarter_projects.json");
const DEFAULT_BROWSERS = Math.max(2, Math.min(6, os.cpus().length));

function printHelp() {
  console.log(`Usage:
  node parse-boomstarter.mjs [options]

Options:
  --links-output       JSON со всеми ссылками проектов (default: ${DEFAULT_LINKS_OUTPUT})
  --projects-output    JSON с распарсенными проектами (default: ${DEFAULT_PROJECTS_OUTPUT})
  --links-input        Взять ссылки из готового JSON и пропустить этап discover
  --browsers, -b       Кол-во параллельных браузеров для парсинга проектов (default: ${DEFAULT_BROWSERS})
  --timeout            Таймаут открытия страницы, мс (default: 60000)
  --retries            Повторы при ошибке (default: 2)
  --save-every         Чекпоинт каждые N проектов (default: 25)
  --max-projects       Ограничить кол-во проектов (для теста)
  --max-discover-clicks Ограничить клики "Загрузить еще" на каждый state (для теста)
  --headless           Запуск без UI (default)
  --headed             Запуск с UI
  --help, -h           Справка`);
}

function parseNumber(value, fallback, min = 0) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < min) {
    return fallback;
  }
  return number;
}

function parseArgs(argv) {
  const args = {
    linksOutput: DEFAULT_LINKS_OUTPUT,
    projectsOutput: DEFAULT_PROJECTS_OUTPUT,
    linksInput: null,
    browsers: DEFAULT_BROWSERS,
    timeoutMs: 60000,
    retries: 2,
    saveEvery: 25,
    maxProjects: null,
    maxDiscoverClicks: null,
    headless: true,
    help: false
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];

    if (arg === "--help" || arg === "-h") {
      args.help = true;
      continue;
    }
    if (arg === "--links-output") {
      args.linksOutput = path.resolve(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === "--projects-output") {
      args.projectsOutput = path.resolve(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === "--links-input") {
      args.linksInput = path.resolve(argv[i + 1]);
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
    if (arg === "--max-projects") {
      args.maxProjects = parseNumber(argv[i + 1], null, 1);
      i += 1;
      continue;
    }
    if (arg === "--max-discover-clicks") {
      args.maxDiscoverClicks = parseNumber(argv[i + 1], null, 1);
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

function cleanText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const cleaned = value.replace(/\u00A0/g, " ").replace(/\s+/g, " ").trim();
  return cleaned || null;
}

function normalizeUrl(value, baseUrl = null) {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim().replaceAll("&amp;", "&");
  if (!trimmed || trimmed.startsWith("javascript:") || trimmed.startsWith("data:")) {
    return null;
  }

  try {
    if (trimmed.startsWith("//")) {
      return new URL(`https:${trimmed}`).toString();
    }
    if (baseUrl) {
      return new URL(trimmed, baseUrl).toString();
    }
    return new URL(trimmed).toString();
  } catch {
    return null;
  }
}

function normalizeProjectUrl(url, baseUrl = null) {
  const normalized = normalizeUrl(url, baseUrl);
  if (!normalized) {
    return null;
  }

  try {
    const parsed = new URL(normalized);
    if (!parsed.hostname.endsWith("boomstarter.ru")) {
      return null;
    }
    const parts = parsed.pathname.split("/").filter(Boolean);
    if (parts[0] !== "projects" || parts.length < 3) {
      return null;
    }
    parsed.hash = "";
    parsed.search = "";
    return parsed.toString().replace(/\/+$/, "");
  } catch {
    return null;
  }
}

function extractUrlsFromText(text, baseUrl = null) {
  if (typeof text !== "string" || !text) {
    return [];
  }

  const urls = [];
  const attrRegex = /(?:href|src|poster|data-src)\s*=\s*["']([^"']+)["']/gi;
  const directRegex = /\bhttps?:\/\/[^\s"'<>]+/gi;
  let match;

  while ((match = attrRegex.exec(text)) !== null) {
    const normalized = normalizeUrl(match[1], baseUrl);
    if (normalized) {
      urls.push(normalized);
    }
  }

  while ((match = directRegex.exec(text)) !== null) {
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
  return /\.(?:jpg|jpeg|png|webp|gif|bmp|svg|avif)(?:$|[?#])/i.test(url);
}

function isVideoUrl(url) {
  if (!url) {
    return false;
  }
  return (
    /\.(?:mp4|mov|webm|m3u8)(?:$|[?#])/i.test(url) ||
    /(youtube\.com|youtu\.be|vimeo\.com|rutube\.ru|vkvideo|\/video\/|twitter:player)/i.test(url)
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

function parseIntFromText(text) {
  const cleaned = cleanText(text);
  if (!cleaned) {
    return null;
  }
  const match = cleaned.match(/(\d[\d\s]*)/);
  if (!match) {
    return null;
  }
  const value = Number(match[1].replace(/\s+/g, ""));
  return Number.isFinite(value) ? value : null;
}

function parsePercent(text) {
  const cleaned = cleanText(text);
  if (!cleaned) {
    return null;
  }
  const match = cleaned.match(/(\d+(?:[.,]\d+)?)\s*%/);
  if (!match) {
    return null;
  }
  const value = Number(match[1].replace(",", "."));
  return Number.isFinite(value) ? value : null;
}

function parseAllIntegers(text) {
  const cleaned = cleanText(text);
  if (!cleaned) {
    return [];
  }
  return [...cleaned.matchAll(/(\d[\d\s]*)/g)]
    .map((match) => Number(match[1].replace(/\s+/g, "")))
    .filter((value) => Number.isFinite(value));
}

function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms <= 0) {
    return "0s";
  }
  const seconds = Math.floor(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const restSeconds = seconds % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m ${restSeconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${restSeconds}s`;
  }
  return `${restSeconds}s`;
}

async function createContext(browser, headlessIgnored = true) {
  return browser.newContext({
    locale: "ru-RU",
    timezoneId: "Europe/Moscow",
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    viewport: { width: 1440, height: 900 }
  });
}

async function detectStates(page) {
  return page.evaluate(() => {
    const radios = Array.from(
      document.querySelectorAll(".control-content__status input[type='radio'].checked_status")
    );
    const states = radios
      .map((radio) => {
        const name = radio.getAttribute("name");
        const label = radio.closest("label")?.querySelector(".ui-checkbox__text")?.textContent?.trim() || name;
        return name ? { name, label } : null;
      })
      .filter(Boolean);
    return states;
  });
}

async function collectVisibleProjectLinks(page) {
  return page.evaluate(() => {
    const normalizeProjectUrlInPage = (href) => {
      if (!href) {
        return null;
      }
      try {
        const url = new URL(href, window.location.origin);
        const parts = url.pathname.split("/").filter(Boolean);
        if (parts[0] !== "projects" || parts.length < 3) {
          return null;
        }
        url.search = "";
        url.hash = "";
        return url.toString().replace(/\/+$/, "");
      } catch {
        return null;
      }
    };

    return [...new Set(
      Array.from(document.querySelectorAll('a[href*="/projects/"]'))
        .map((anchor) => normalizeProjectUrlInPage(anchor.getAttribute("href") || anchor.href))
        .filter(Boolean)
    )];
  });
}

async function hasShowMore(page) {
  const button = page.locator("a.main-project-list__show-more, a:has-text('Загрузить еще')").first();
  return (await button.count()) > 0;
}

async function clickShowMore(page, timeoutMs) {
  const button = page.locator("a.main-project-list__show-more, a:has-text('Загрузить еще')").first();
  if ((await button.count()) === 0) {
    return false;
  }

  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      await button.scrollIntoViewIfNeeded().catch(() => {});
      await Promise.allSettled([
        page.waitForLoadState("networkidle", { timeout: Math.min(5000, timeoutMs) }),
        button.click({ timeout: 12000, force: true })
      ]);
      await page.waitForTimeout(700);
      return true;
    } catch {
      if (attempt === 2) {
        return false;
      }
      await page.waitForTimeout(600);
    }
  }
  return false;
}

function resolveStateUrl(stateName) {
  if (stateName === "top") {
    return "https://boomstarter.ru/discover/most-funded?v2=true";
  }
  return `https://boomstarter.ru/discover?v2=true&scope=${encodeURIComponent(stateName)}`;
}

async function discoverProjectLinks(args) {
  const browser = await chromium.launch({ headless: args.headless });
  const context = await createContext(browser);
  const page = await context.newPage();

  await page.goto(DISCOVER_SEED_URL, { waitUntil: "domcontentloaded", timeout: args.timeoutMs });
  await page.waitForTimeout(2500);

  const states = await detectStates(page);
  if (!states.length) {
    throw new Error("Не удалось получить список state фильтров на странице discover.");
  }

  const perState = [];
  const byProjectStateMap = new Map();

  for (const state of states) {
    const stateUrl = resolveStateUrl(state.name);
    await page.goto(stateUrl, { waitUntil: "domcontentloaded", timeout: args.timeoutMs });
    await page.waitForTimeout(2500);

    let links = await collectVisibleProjectLinks(page);
    let stagnantClicks = 0;
    let clicks = 0;

    while (await hasShowMore(page)) {
      clicks += 1;
      const previousCount = links.length;
      const clicked = await clickShowMore(page, args.timeoutMs);
      if (!clicked) {
        break;
      }

      links = unique([...links, ...(await collectVisibleProjectLinks(page))]);
      if (links.length === previousCount) {
        stagnantClicks += 1;
      } else {
        stagnantClicks = 0;
      }

      if (stagnantClicks >= 3 || clicks >= 500) {
        break;
      }
      if (args.maxDiscoverClicks && clicks >= args.maxDiscoverClicks) {
        break;
      }
      if (clicks % 5 === 0) {
        console.log(`  ${state.name}: клик ${clicks}, собрано ${links.length}`);
      }
    }

    const normalizedLinks = unique(links.map((link) => normalizeProjectUrl(link)).filter(Boolean));
    for (const link of normalizedLinks) {
      if (!byProjectStateMap.has(link)) {
        byProjectStateMap.set(link, new Set());
      }
      byProjectStateMap.get(link).add(state.name);
    }

    perState.push({
      state: state.name,
      label: state.label,
      sourceUrl: stateUrl,
      totalProjects: normalizedLinks.length,
      projectUrls: normalizedLinks
    });

    console.log(`State ${state.name}: ${normalizedLinks.length} ссылок`);
  }

  await page.close();
  await context.close();
  await browser.close();

  const allProjectUrls = unique(perState.flatMap((item) => item.projectUrls));
  const projectStates = allProjectUrls.map((url) => ({
    url,
    states: [...(byProjectStateMap.get(url) || [])]
  }));

  return {
    generatedAt: new Date().toISOString(),
    discoverSeedUrl: DISCOVER_SEED_URL,
    states: perState,
    totalUniqueProjects: allProjectUrls.length,
    projectUrls: allProjectUrls,
    projectStates
  };
}

async function readLinksFile(linksInputPath) {
  if (!existsSync(linksInputPath)) {
    throw new Error(`Файл ссылок не найден: ${linksInputPath}`);
  }
  const parsed = JSON.parse(await fs.readFile(linksInputPath, "utf8"));
  const urls = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed.projectUrls)
      ? parsed.projectUrls
      : [];

  const normalizedUrls = unique(urls.map((url) => normalizeProjectUrl(url)).filter(Boolean));
  const stateMap = new Map();

  if (Array.isArray(parsed.projectStates)) {
    for (const item of parsed.projectStates) {
      const normalized = normalizeProjectUrl(item?.url);
      if (!normalized) {
        continue;
      }
      const states = Array.isArray(item.states) ? unique(item.states.map((state) => cleanText(state))) : [];
      stateMap.set(normalized, states.filter(Boolean));
    }
  }

  return {
    generatedAt: new Date().toISOString(),
    discoverSeedUrl: parsed.discoverSeedUrl || null,
    states: Array.isArray(parsed.states) ? parsed.states : [],
    totalUniqueProjects: normalizedUrls.length,
    projectUrls: normalizedUrls,
    projectStates: normalizedUrls.map((url) => ({ url, states: stateMap.get(url) || [] }))
  };
}

async function scrapeProjectPage(page, projectUrl, timeoutMs) {
  await page.goto(projectUrl, { waitUntil: "domcontentloaded", timeout: timeoutMs });
  await page.waitForTimeout(1200);

  const extracted = await page.evaluate(() => {
    const clean = (value) =>
      typeof value === "string" ? value.replace(/\u00A0/g, " ").replace(/\s+/g, " ").trim() || null : null;
    const readText = (selector) => clean(document.querySelector(selector)?.textContent || "");
    const readAttr = (selector, attr) => document.querySelector(selector)?.getAttribute(attr) || null;
    const toAbs = (value) => {
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
    const uniq = (array) => [...new Set(array.filter(Boolean))];
    const countFromAnchor = (hrefPart) => {
      const anchor = document.querySelector(`.menu.js-tabs a[href*="${hrefPart}"]`);
      if (!anchor) {
        return null;
      }
      const countNode = anchor.querySelector(".count");
      const text = clean(countNode?.textContent || anchor.textContent || "");
      if (!text) {
        return 0;
      }
      const match = text.match(/(\d[\d\s]*)/);
      return match ? Number(match[1].replace(/\s+/g, "")) : 0;
    };
    const descriptionNode =
      document.querySelector(".content.js-tabs-content.js-active .description") ||
      document.querySelector(".content.js-tabs-content .description");
    const descriptionHtml = descriptionNode ? descriptionNode.innerHTML : null;
    const descriptionText = clean(descriptionNode?.textContent || "");
    const meta = {
      description: document.querySelector('meta[name="description"]')?.getAttribute("content") || null,
      ogTitle: document.querySelector('meta[property="og:title"]')?.getAttribute("content") || null,
      ogDescription: document.querySelector('meta[property="og:description"]')?.getAttribute("content") || null,
      ogImage: document.querySelector('meta[property="og:image"]')?.getAttribute("content") || null,
      ogUrl: document.querySelector('meta[property="og:url"]')?.getAttribute("content") || null,
      twitterPlayer: document.querySelector('meta[property="twitter:player"]')?.getAttribute("content") || null,
      twitterPlayerStream:
        document.querySelector('meta[property="twitter:player:stream"]')?.getAttribute("content") || null,
      twitterPlayerStreamContentType:
        document.querySelector('meta[property="twitter:player:stream:content_type"]')?.getAttribute("content") || null
    };
    const allAnchorLinks = uniq(
      Array.from(document.querySelectorAll("a[href]"))
        .map((anchor) => toAbs(anchor.getAttribute("href") || anchor.href))
        .filter(Boolean)
    );
    const domImages = uniq(
      Array.from(document.querySelectorAll("img"))
        .map((image) => toAbs(image.currentSrc || image.src || image.getAttribute("src")))
        .filter(Boolean)
    );
    const domVideos = uniq([
      ...Array.from(document.querySelectorAll("video"))
        .map((video) => toAbs(video.currentSrc || video.src || video.getAttribute("src")))
        .filter(Boolean),
      ...Array.from(document.querySelectorAll("video source"))
        .map((source) => toAbs(source.src || source.getAttribute("src")))
        .filter(Boolean),
      ...Array.from(document.querySelectorAll("iframe"))
        .map((frame) => toAbs(frame.src || frame.getAttribute("src")))
        .filter(Boolean)
    ]);

    const tabCounts = {
      newsCount: countFromAnchor("/posts"),
      commentsCount: countFromAnchor("/comments"),
      backersCount: countFromAnchor("/backers")
    };

    if (tabCounts.newsCount === null && document.querySelector(".js-news .no-posts")) {
      tabCounts.newsCount = 0;
    }
    if (tabCounts.commentsCount === null) {
      const commentsList = document.querySelector(".comments-list");
      if (commentsList && commentsList.children.length === 0) {
        tabCounts.commentsCount = 0;
      }
    }

    return {
      finalUrl: window.location.href.replace(/\/+$/, ""),
      title: readText("h1.title"),
      subtitle: readText(".blurb"),
      location: readText(".main-extra-info .location"),
      category: readText(".main-extra-info .category"),
      creatorName: readText(".creator .name"),
      creatorAvatar: toAbs(readAttr(".creator .avatar img", "src")),
      creatorProjectsInfo: readText(".creator .achivements-v2 span"),
      mainImage: toAbs(readAttr(".video-wrapper img", "src")),
      fundedNowText: readText(".money-backed .backed-now"),
      fundedTargetText: readText(".money-backed .backed-target"),
      backersText: readText(".backers-now"),
      timeLeftText: readText(".time-left"),
      progressPercentText: readText(".progress .percentage"),
      timerText: readText(".progress .timer"),
      canonicalUrl: readAttr('link[rel="canonical"]', "href"),
      pageTitle: document.title || null,
      descriptionHtml,
      descriptionText,
      tabCounts,
      meta,
      allAnchorLinks,
      domImages,
      domVideos,
      gonProject: window.gon?.Project || null,
      gonRewards: Array.isArray(window.gon?.Rewards) ? window.gon.Rewards : []
    };
  });

  if (!extracted.title) {
    throw new Error("Не удалось получить заголовок проекта.");
  }
  return extracted;
}

function mapRewards(rewards, baseUrl) {
  if (!Array.isArray(rewards)) {
    return [];
  }

  return rewards.map((reward) => ({
    id: reward?.id ?? null,
    title: reward?.title ?? null,
    description: reward?.description ?? null,
    imageUrl: normalizeUrl(reward?.photo, baseUrl),
    amount: typeof reward?.amount === "number" ? reward.amount : parseIntFromText(String(reward?.amount || "")),
    limit: typeof reward?.limit === "number" ? reward.limit : parseIntFromText(String(reward?.limit || "")),
    backingsCount:
      typeof reward?.backings_count === "number"
        ? reward.backings_count
        : parseIntFromText(String(reward?.backings_count || "")),
    estimatedDelivery: cleanText(reward?.estimated_delivery),
    delivery: cleanText(reward?.delivery),
    isDeliverable: reward?.is_deliverable ?? null,
    isPopular: reward?.is_popular ?? null,
    isSoldOut: reward?.is_sold_out ?? null,
    isActive: reward?.is_active ?? null,
    rewardPath: normalizeUrl(reward?.reward_path, baseUrl)
  }));
}

function buildProjectRecord(projectUrl, extracted, stateMapEntry, scrapedAtIso) {
  const finalUrl = normalizeProjectUrl(extracted.finalUrl || projectUrl) || normalizeProjectUrl(projectUrl);
  const fundedNow = parseIntFromText(extracted.fundedNowText);
  const targetCandidates = parseAllIntegers(extracted.fundedTargetText);
  const fundedTarget = targetCandidates.length ? targetCandidates[0] : null;
  const backersFromMoney = parseIntFromText(extracted.backersText);
  const progressPercent = parsePercent(extracted.progressPercentText);

  const rewards = mapRewards(extracted.gonRewards, finalUrl);
  const rewardsFromDescriptions = unique(rewards.flatMap((reward) => extractUrlsFromText(reward.description || "", finalUrl)));
  const descriptionUrls = unique(extractUrlsFromText(extracted.descriptionHtml || "", finalUrl));
  const descriptionClassified = classifyUrls(descriptionUrls);
  const rewardsDescClassified = classifyUrls(rewardsFromDescriptions);

  const rewardImages = unique(rewards.map((reward) => reward.imageUrl));
  const imageUrls = unique([
    normalizeUrl(extracted.mainImage, finalUrl),
    normalizeUrl(extracted.meta?.ogImage, finalUrl),
    ...rewardImages,
    ...descriptionClassified.imageUrls,
    ...(extracted.domImages || []).map((url) => normalizeUrl(url, finalUrl))
  ]);

  const videoUrls = unique([
    normalizeUrl(extracted.meta?.twitterPlayer, finalUrl),
    normalizeUrl(extracted.meta?.twitterPlayerStream, finalUrl),
    ...descriptionClassified.videoUrls,
    ...(extracted.domVideos || []).map((url) => normalizeUrl(url, finalUrl))
  ]);

  const externalLinks = unique([
    ...(extracted.allAnchorLinks || []).map((url) => normalizeUrl(url, finalUrl)),
    ...descriptionClassified.otherUrls,
    ...rewardsDescClassified.otherUrls
  ]).filter((url) => !imageUrls.includes(url) && !videoUrls.includes(url));

  const tabCounts = extracted.tabCounts || {};
  const participantsCount =
    typeof tabCounts.backersCount === "number" ? tabCounts.backersCount : backersFromMoney;

  const dateLikeStrings = unique(
    [extracted.timerText, extracted.timeLeftText]
      .filter(Boolean)
      .flatMap((value) => value.match(/\d{1,2}\s+[а-яё]+(?:\s+\d{4})?/gi) || [])
  );

  return {
    status: "ok",
    sourceUrl: projectUrl,
    finalUrl,
    discoveredStates: stateMapEntry || [],
    scrapedAt: scrapedAtIso,
    pageTitle: extracted.pageTitle || null,
    canonicalUrl: normalizeUrl(extracted.canonicalUrl, finalUrl),
    card: {
      projectId: extracted.gonProject?.id ?? null,
      creatorId: extracted.gonProject?.creator_id ?? null,
      projectType: extracted.gonProject?.type ?? null,
      title: extracted.title || null,
      subtitle: extracted.subtitle || null,
      location: extracted.location || null,
      category: extracted.category || null,
      creatorName: extracted.creatorName || null,
      creatorAvatar: normalizeUrl(extracted.creatorAvatar, finalUrl),
      creatorProjectsInfo: extracted.creatorProjectsInfo || null,
      fundedNow,
      fundedTarget,
      fundedNowText: extracted.fundedNowText || null,
      fundedTargetText: extracted.fundedTargetText || null,
      progressPercent,
      progressPercentText: extracted.progressPercentText || null,
      backersText: extracted.backersText || null,
      timeLeftText: extracted.timeLeftText || null,
      timerText: extracted.timerText || null,
      dateLikeStrings,
      mainImageUrl: normalizeUrl(extracted.mainImage, finalUrl)
    },
    description: {
      html: extracted.descriptionHtml || null,
      text: stripHtml(extracted.descriptionHtml || extracted.descriptionText || "")
    },
    counts: {
      newsCount: typeof tabCounts.newsCount === "number" ? tabCounts.newsCount : null,
      commentsCount: typeof tabCounts.commentsCount === "number" ? tabCounts.commentsCount : null,
      participantsCount: typeof participantsCount === "number" ? participantsCount : null
    },
    rewards: {
      totalRewards: rewards.length,
      items: rewards
    },
    media: {
      imageUrls,
      videoUrls,
      externalLinks
    },
    meta: extracted.meta || null,
    raw: {
      gonProject: extracted.gonProject || null,
      gonRewards: extracted.gonRewards || []
    }
  };
}

async function scrapeProjects(urls, projectStateMap, args) {
  const results = new Array(urls.length);
  const workerCount = Math.max(1, Math.min(args.browsers, urls.length));
  const startedAt = Date.now();
  const checkpointPath = `${args.projectsOutput}.checkpoint.json`;

  let cursor = 0;
  let completed = 0;
  let success = 0;
  let failed = 0;
  let lastCheckpointCount = 0;
  let checkpointChain = Promise.resolve();

  const getTask = () => {
    if (cursor >= urls.length) {
      return null;
    }
    const index = cursor;
    cursor += 1;
    return { index, url: urls[index] };
  };

  const snapshot = (final = false) => ({
    generatedAt: new Date().toISOString(),
    final,
    input: {
      totalUrls: urls.length
    },
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

  const scheduleCheckpoint = (force = false) => {
    if (!force && completed - lastCheckpointCount < args.saveEvery) {
      return;
    }
    lastCheckpointCount = completed;
    checkpointChain = checkpointChain
      .then(() => fs.writeFile(checkpointPath, JSON.stringify(snapshot(false), null, 2), "utf8"))
      .catch((error) => {
        console.error(`Не удалось сохранить checkpoint: ${error.message}`);
      });
  };

  const logStep = Math.max(10, Math.floor(urls.length / 200));

  const worker = async (workerId) => {
    const browser = await chromium.launch({ headless: args.headless });
    const context = await createContext(browser);
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
      const task = getTask();
      if (!task) {
        break;
      }

      let extracted = null;
      let errorMessage = null;

      for (let attempt = 1; attempt <= args.retries + 1; attempt += 1) {
        try {
          extracted = await scrapeProjectPage(page, task.url, args.timeoutMs);
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
        const states = projectStateMap.get(normalizeProjectUrl(task.url) || task.url) || [];
        results[task.index] = buildProjectRecord(task.url, extracted, states, nowIso);
        success += 1;
      } else {
        results[task.index] = {
          status: "error",
          sourceUrl: task.url,
          finalUrl: null,
          discoveredStates: projectStateMap.get(normalizeProjectUrl(task.url) || task.url) || [],
          scrapedAt: nowIso,
          error: errorMessage || "Unknown error"
        };
        failed += 1;
        console.error(`[worker ${workerId}] ${task.url} -> ${errorMessage}`);
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

      scheduleCheckpoint(false);
    }

    await page.close();
    await context.close();
    await browser.close();
  };

  await Promise.all(Array.from({ length: workerCount }, (_, index) => worker(index + 1)));
  scheduleCheckpoint(true);
  await checkpointChain;

  const finalOutput = snapshot(true);
  await fs.writeFile(args.projectsOutput, JSON.stringify(finalOutput, null, 2), "utf8");
  if (existsSync(checkpointPath)) {
    await fs.unlink(checkpointPath).catch(() => {});
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }

  let linksData;
  if (args.linksInput) {
    console.log(`Читаю ссылки из ${args.linksInput}`);
    linksData = await readLinksFile(args.linksInput);
  } else {
    console.log("Собираю ссылки проектов по всем state фильтрам...");
    linksData = await discoverProjectLinks(args);
    await fs.writeFile(args.linksOutput, JSON.stringify(linksData, null, 2), "utf8");
    console.log(`Ссылки сохранены в ${args.linksOutput}`);
  }

  let projectUrls = linksData.projectUrls;
  if (args.maxProjects) {
    projectUrls = projectUrls.slice(0, args.maxProjects);
  }

  if (!projectUrls.length) {
    throw new Error("Не найдено ни одной ссылки проекта для парсинга.");
  }

  const projectStateMap = new Map(
    (linksData.projectStates || []).map((item) => [normalizeProjectUrl(item.url) || item.url, item.states || []])
  );

  console.log(`Проектов к парсингу: ${projectUrls.length}`);
  console.log(`Параллельных браузеров: ${Math.max(1, Math.min(args.browsers, projectUrls.length))}`);
  console.log(`Выходной файл: ${args.projectsOutput}`);

  await scrapeProjects(projectUrls, projectStateMap, args);
  console.log("Готово.");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exitCode = 1;
});
