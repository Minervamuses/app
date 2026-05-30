# C1 工具路由評測 — 首次跑況與發現

日期:2026-05-30
指令:`python -m agent.cli.eval --claim c1 --split dev --allow-skips --no-mcp`
run_id:`c1-20260530T080146Z-c36d63e0`

## 跑況總覽

- dev 共 8 題:**評到 5、跳過 3**(3 個 web 題因 `--no-mcp` 沒載入 web 工具而 skip)。
- `routing_accuracy = 0.6`(評到的 5 題中 3 題全項通過)。
- **此 run 不可作為正式 baseline**:`baseline_eligible=false`、`skipped=3`、`allow_skips=true`。
  有 skip 的 run 不能跟「評滿 8 題」的 run 直接比。

## 分項數字(比總分有用)

| metric | 值 | 解讀 |
|---|---|---|
| first_tool_accuracy | 1.00 | 第一手永遠選對 |
| tool_family_accuracy | 1.00 | 工具家族永遠選對 |
| no_tool_accuracy | 1.00 | 該閉嘴時閉嘴 |
| tools_coverage | 0.75 | 偶爾不去碰該用的專門工具 |
| forbidden_tool_accuracy | 0.80 | 出現一次越界 |
| tool_count_accuracy | 0.60 | 最弱:愛迴圈、叫過頭 |
| routing_accuracy | 0.60 | 總分(全項過才算過) |

## 兩個真實發現

1. **rag_search(「scoring 模組怎麼運作」)— 路由對,但話太多**
   - 該叫 1–4 次,實際叫了 5 次(rag_explore + 4×rag_search)。唯一掛在 count。

2. **rag_get_context(「embedding 模組」+「展開上下文」追問)— 爆走**
   - 叫了 **15 次**工具、狂刷 rag_search 12 次、**逃去叫 `bash`**,且**全程沒用 rag_get_context**。
   - 三檢查同掛:count(15≫6)、forbidden(碰 bash)、tools_covered(沒用到 rag_get_context)。
   - 行為畫面:面對「展開附近內容」,不會用專門工具,而是土法暴力狂搜、甚至逃去 shell。

## 結論

agent **「選對門」的直覺很穩**(first_tool / family / no_tool 全 1.0),
但有兩個傾向:**(a) 愛迴圈、叫過頭**(count 最差),**(b) 不愛用專門工具**
(該 rag_get_context 卻狂刷 rag_search,甚至逃 bash)。

## 待辦

- [ ] 收 recursion limit / prompt,壓掉「叫過頭、逃去 bash」的爆走(主因疑為迴圈控制)。
- [ ] 載入 MCP web 工具、評滿 8 題、不加 `--allow-skips`,取得**正式 c1 baseline**。
- [ ] 接 temp=0 + n-sample 多跑報 mean±std(現為單跑快照,count 數字會抖)。
- [ ] 追查 embedding 追問為何升級到 bash —— 看是否 recursion limit 沒收好。

## 註

- 數字非決定性:主模型 temp=0.3、無 seed,「叫 15 次」會隨 run 變動。
- 此 run 走真 agent + 真檢索(本地 Ollama bge-m3 嵌入),故耗時明顯高於 c3。
