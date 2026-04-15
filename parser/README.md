# Planeta Parser (Playwright)

Параллельный парсер карточек проектов `planeta.ru/campaigns/*`.

Собирает:
- заголовок, подзаголовок, статус, суммы, даты, регион, категорию, автора, ссылки карточки;
- описание проекта (`html` + `text`);
- все вознаграждения;
- количество новостей, комментариев и участников;
- ссылки на картинки, видео и внешние ссылки;
- сырые блоки данных из `window.INITIAL_STATE`.

## Установка

```bash
npm install
npx playwright install chromium
```

## Запуск

По умолчанию вход: `/Users/zhdanovmaxim/Downloads/ссылки планета.json`  
По умолчанию выход: `./planeta_projects.json`

```bash
npm run parse
```

Пример с явными параметрами:

```bash
node parse-planeta.mjs \
  --input "/Users/zhdanovmaxim/Downloads/ссылки планета.json" \
  --output "./planeta_projects.json" \
  --browsers 6 \
  --retries 2 \
  --timeout 45000 \
  --save-every 50
```

## Полезно

- Для MacBook Pro M3 Pro обычно хорошо: `--browsers 6` (можно поднять до `8`).
- Во время длинного прогона пишется checkpoint: `planeta_projects.json.checkpoint.json`.
- В итоге: `status: "ok"` для успешных и `status: "error"` для проблемных URL.

## Boomstarter

Двухэтапный парсер:
1. Сначала собирает все ссылки проектов из `discover` по всем `state` (`in_progress`, `ended`, `top`).
2. Потом параллельно парсит каждую страницу проекта.

Скрипт:

```bash
npm run parse:boomstarter
```

По умолчанию создаёт:
- `./boomstarter_links.json` — список ссылок из всех state;
- `./boomstarter_projects.json` — итоговый парсинг проектов.

Пример запуска:

```bash
node parse-boomstarter.mjs \
  --links-output "./boomstarter_links.json" \
  --projects-output "./boomstarter_projects.json" \
  --browsers 6 \
  --retries 2 \
  --timeout 60000 \
  --save-every 25
```

Тестовый прогон:

```bash
node parse-boomstarter.mjs --max-projects 20
```

Быстрый smoke-тест (ограничить и discover, и парсинг):

```bash
node parse-boomstarter.mjs --max-discover-clicks 3 --max-projects 20
```
