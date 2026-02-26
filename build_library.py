"""
建立歌曲資料庫 (song_library.csv)
掃描 songs/ 資料夾中的 .wav 檔案

注意：已移除 CLAP embedding，現在只建立歌曲清單。
特徵提取由 extract_features.py 負責。
"""
import os
import pandas as pd

SONGS_FOLDER = "./songs"

songs = []
for filename in sorted(os.listdir(SONGS_FOLDER)):
    if filename.endswith((".wav", ".mp3")):
        songs.append({
            "filename": filename,
            "title": os.path.splitext(filename)[0],
        })
        print(f"  ✅ {filename}")

pd.DataFrame(songs).to_csv("song_library.csv", index=False)
print(f"\n✅ 完成！共 {len(songs)} 首歌寫入 song_library.csv")