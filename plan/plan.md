你阅读下 my_video/cli/commands/subtitle.py 的流程 大致是 补上标点符号->校对优化->划分->按每一句翻译->生成字幕

但是我有时候会从油管上下载来字幕，可能是srt格式，也可能是vtt格式
所以如果status.json的origin/subtitle_path，多了如下步骤
1. 通过ASRData.from_subtitle_file 获取ASRData对象（油管字幕版）
2. 通过ASRData.from_whisperx_json 获取ASRData对象（whisperx版）
3. 对比ASRData.to_txt得到的字符串
    请注意需要做下预先处理，把连续重复的空格给替换成单个空格
    大致逻辑可以参考my_video/core/optimize/optimize.py的_build_token_opcodes
    比如油管字幕版是 Let's try plugging in the USB charger. It charges normally. Let's try turning on the console
    whisperx版是   Let's try plus in the USB. It charges normally. Let's try turning turning on the console.
    那么我希望能输出这么个字符串，以油管字幕版为准，让我看看是哪里发生了变化
    Let's try 【plus/plugging】 in the USB 【∅/charger】. It charges normally. Let's try turning 【turning/∅】 on the console
    【plus/plugging】，【∅/charger】，【turning/∅】 这三个分别代表 替换， 新增，删除
4. 将对比结果，也就是第3步的结果保存到 paths.compare_txt 里
5. 终端输出“以whisperx(1)为准，还是自带字幕(2)为准？”，要求用户输入1或2
    如果选1,那就是whisperx的为准。如果选2,这里还需要额外处理，把油管字幕版回写到whisperx版ASRData，因为whisperx版ASRData是时间级别时间戳
    回写的逻辑可以参考_rewrite_group_segments
    这里注意是否需要把共同逻辑抽离成文件，方便重用

后面就是和status.json的origin/subtitle_path为空共同的逻辑
终端输出“是否需要llm校对标点符号？ Y/N?” 如果用户选Y，执行补上标点符号，把对比结果输出到paths.punctuation_txt，反之跳过
终端输出“是否需要llm校对优化？ Y/N?” 如果用户选Y执行校对优化，把对比结果输出到paths.optimized_txt，反之跳过
划分
按每一句翻译
生成字幕

PS: 题外话，_build_group_change_logs也需要重构，按照Let's try 【plus/plugging】 in the USB 【∅/charger】. It charges normally. Let's try turning 【turning/∅】 on the console这样的格式来生成，同样以优化版的为准。
ASRSentenceData的optimize_logs需要更改名字为optimize_log，字符串格式。to_txt理论上是每一句换行输出，这里注意每一句如果optimize_log存在，就直接用optimize_log来输出
