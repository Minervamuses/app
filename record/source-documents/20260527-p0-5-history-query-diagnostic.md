# P0.5 history query diagnostic

Date: 2026-05-25
Branch: `fix/history-tool-availability`

## Collection access

- `persist_dir`: `/home/minervamuses/PiDNA2/rag/store`
- `chat_history` directory: `/home/minervamuses/PiDNA2/rag/store/chat_history`
- `chroma.sqlite3`: present
- Collection document count by direct inspection: 108
- Oldest timestamp by direct inspection: `2026-04-25T17:11:26.806479+00:00`
- Newest timestamp by direct inspection: `2026-05-24T12:25:03.060958+00:00`

## recall_history query styles

1. Semantic description, Chinese
   - Query: `我一月上半做的研究成果`
   - Result count: 5
   - Sample hit: user turn 1 at `2026-05-24T12:13:20.671238+00:00`, text begins `我一月上半部的成果如果要寫成論文...`

2. Semantic description, English
   - Query: `early January research progress`
   - Result count: 5
   - Sample hit: user turn 5 at `2026-04-26T04:15:47.601992+00:00`, text begins `極簡回答：一月中我在忙什麼？`

3. Time anchor plus topic, Chinese
   - Query: `一月 人工智慧`
   - Result count: 5
   - Sample hit: user turn 4 at `2026-05-24T12:22:30.217762+00:00`, text begins `1.人工智慧。2.你自行去看一下應該有紀錄`

4. Time anchor plus topic, English
   - Query: `AI January experiments`
   - Result count: 5
   - Sample hit: user turn 1 at `2026-05-24T11:42:47.173514+00:00`, text begins `我1月上半月的成果如果要寫成paper...`

5. Role filter, semantic description
   - Query: `我一月上半做的研究成果`, `role=assistant`
   - Result count: 5
   - Sample hit: assistant turn 3 at `2026-05-24T12:21:15.415036+00:00`, text begins `1. 請提供研究的主題或學科/子領域...`

6. Role filter, time anchor plus topic
   - Query: `一月 人工智慧`, `role=assistant`
   - Result count: 5
   - Sample hit: assistant turn 3 at `2026-05-24T12:21:15.415036+00:00`, text begins `1. 請提供研究的主題或學科/子領域...`

7. Bulk inspection
   - Query: `conversation history`, `k=20`
   - Result count: 20
   - Sample hit: user turn 2 at `2026-04-25T17:14:21.941459+00:00`

## Conclusion

`hit`. The local `chat_history` collection is accessible, non-empty, and multiple query styles return January-related results. P0 unblocks a real retrieval path for the problem scenario, though retrieved assistant hits also show previous turns where the agent asked for intake details instead of using the available history.
