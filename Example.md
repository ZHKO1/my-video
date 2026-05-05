uv run yt-dlp -f bestvideo+bestaudio/best
--no-playlist
--cookies ./cookies.txt -vU --proxy socks5://192.168.71.5:20170/ --remote-components ejs:github https://www.youtube.com/watch?v=qCEZC3cPc1s


uv run yt-dlp \
    --skip-download \
  --remote-components ejs:github \
  --proxy socks5://192.168.3.6:2222/ \
  --cookies ./cookies.txt \
  --write-subs \
  --sub-langs "en.*" \
  --sub-format srt \
                --no-embed-subs \
                -o a.srt \
  "https://www.youtube.com/watch?v=u-ahOATO62U"


uv run yt-dlp --list-subs --proxy socks5://192.168.71.5:20170/ "https://www.youtube.com/watch?v=qCEZC3cPc1s"

uv run yt-dlp --skip-download --write-auto-subs \
  --sub-langs "en" \
  --convert-subs srt \
  --proxy socks5://192.168.71.5:20170/ \
  "https://www.youtube.com/watch?v=qCEZC3cPc1s"