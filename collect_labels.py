"""
Timbre 標註收集工具
用來收集用戶對推薦結果的反饋，改善推薦品質
"""
import pandas as pd
import os
from recommend_v2 import recommend, translate_to_english, song_library

LABELS_FILE = "labels.csv"

# 載入已有的標註
if os.path.exists(LABELS_FILE):
    labels_df = pd.read_csv(LABELS_FILE)
else:
    labels_df = pd.DataFrame(
        columns=["mood", "mood_english", "song_title", "label", "comment"])


def collect_session():
    print("=== Timbre 標註收集 ===")
    print("輸入情緒描述，對每首推薦歌曲評分（1=對，0=不對）")
    print("輸入 'quit' 結束\n")

    new_labels = []

    while True:
        mood = input("情緒描述：").strip()
        if mood.lower() == 'quit':
            break

        english = translate_to_english(mood)
        results = recommend(mood, top_k=5, return_results=True)

        for rank, (idx, score) in enumerate(results):
            title = song_library.iloc[idx]['title']
            print(f"\n  [{rank + 1}] {title}  (分數: {score:.3f})")

            while True:
                label_input = input("  對不對？(1/0/s=skip): ").strip()
                if label_input in ['0', '1', 's']:
                    break

            if label_input == 's':
                continue

            comment = input("  簡短評語（可空白）：").strip()

            new_labels.append({
                "mood": mood,
                "mood_english": english,
                "song_title": title,
                "label": int(label_input),
                "comment": comment
            })

    # 存檔
    if new_labels:
        new_df = pd.DataFrame(new_labels)
        labels_df_updated = pd.concat([labels_df, new_df], ignore_index=True)
        labels_df_updated.to_csv(LABELS_FILE, index=False)
        print(
            f"\n✅ 新增 {len(new_labels)} 筆，總共 {len(labels_df_updated)} 筆標註")


collect_session()