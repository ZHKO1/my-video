uv run yt-dlp -f bestvideo+bestaudio/best --cookies ./cookies.txt  --no-playlist -vU --proxy socks5://192.168.71.5:20170/ --remote-components ejs:github https://www.youtube.com/watch?v=qCEZC3cPc1s


uv run yt-dlp \
    --skip-download \
  --remote-components ejs:github \
  --proxy socks5://192.168.71.5:20170/ \
  --cookies ./cookies.txt \
  --write-subs --write-auto-subs \
  --sub-langs "en.*,zh-Hans" \
  --sub-format srt \
  "https://www.youtube.com/watch?v=qCEZC3cPc1s"


uv run yt-dlp --list-subs --proxy socks5://192.168.71.5:20170/ "https://www.youtube.com/watch?v=qCEZC3cPc1s"

uv run yt-dlp --skip-download --write-auto-subs \
  --sub-langs "en" \
  --convert-subs srt \
  --proxy socks5://192.168.71.5:20170/ \
  "https://www.youtube.com/watch?v=qCEZC3cPc1s"