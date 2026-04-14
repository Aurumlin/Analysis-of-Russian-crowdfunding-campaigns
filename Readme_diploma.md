# Словарь данных

## Planeta — `projects_df`

| Столбец                                 | Описание                                      |
| --------------------------------------- | --------------------------------------------- |
| `project_key`                           | Уникальный ключ проекта                       |
| `status`                                | Статус парсинга записи                        |
| `sourceUrl`                             | Исходная ссылка проекта                       |
| `pageTitle`                             | HTML title страницы                           |
| `card.campaignId`                       | ID кампании                                   |
| `card.title`                            | Название проекта                              |
| `card.subtitle`                         | Подзаголовок                                  |
| `card.status`                           | Статус проекта                                |
| `card.targetStatus`                     | Статус достижения цели                        |
| `card.collectedAmount.currencyCode`     | Валюта собранных средств                      |
| `card.collectedAmount.value`            | Сумма собранных средств                       |
| `card.targetAmount.currencyCode`        | Валюта цели                                   |
| `card.targetAmount.value`               | Целевая сумма                                 |
| `card.purchaseCount`                    | Число покупок / поддержек                     |
| `card.daysToFinish`                     | Дней до завершения                            |
| `card.startAt`                          | Дата начала проекта                           |
| `card.finishAt`                         | Дата окончания проекта                        |
| `card.region`                           | Регион проекта                                |
| `card.mainCategory.tagName`             | Категория проекта                             |
| `card.mainCategory.mnemonicName`        | Системное имя категории                       |
| `card.author.id`                        | ID автора                                     |
| `card.author.fid`                       | Строковый ID автора                           |
| `card.author.campaignsAmount`           | Число проектов автора                         |
| `card.author.allCampaignsPurchaseCount` | Суммарные покупки у автора                    |
| `card.links.vk_url`                     | Ссылка на VK                                  |
| `card.links.telegram_url`               | Ссылка на Telegram                            |
| `card.links.author_site_url`            | Сайт автора                                   |
| `description.text`                      | Текстовое описание проекта                    |
| `counts.newsCount`                      | Число новостей                                |
| `counts.commentsCount`                  | Число комментариев                            |
| `counts.participantsCount`              | Число участников                              |
| `counts.purchasesCount`                 | Число покупок                                 |
| `meta.description`                      | Meta description страницы                     |
| `rewards.totalRewards`                  | Число наград                                  |
| `card.hasImage`                         | Есть ли основное изображение                  |
| `image_count`                           | Количество изображений                        |
| `video_count`                           | Количество видео                              |
| `reward_count`                          | Количество наград в таблице rewards           |
| `external_link_count`                   | Количество внешних ссылок                     |
| `funding_ratio`                         | Доля сбора от цели                            |
| `is_successful`                         | Собрана ли цель                               |
| `campaign_duration_days`                | Длительность кампании в днях                  |
| `has_both_dates`                        | Есть ли дата начала и конца                   |
| `title_len_chars`                       | Длина заголовка в символах                    |
| `subtitle_len_chars`                    | Длина подзаголовка в символах                 |
| `description_len_chars`                 | Длина описания в символах                     |
| `meta_description_len_chars`            | Длина meta description в символах             |
| `title_word_count`                      | Число слов в заголовке                        |
| `subtitle_word_count`                   | Число слов в подзаголовке                     |
| `description_word_count`                | Число слов в описании                         |
| `meta_description_word_count`           | Число слов в meta description                 |
| `creator_projects_count`                | Число проектов автора                         |
| `description_has_link_word`             | Есть ли в описании слова про внешние площадки |
