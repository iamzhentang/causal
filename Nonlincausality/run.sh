eval "$(conda shell.bash hook)"
python tune.py anhui > log_anhui 2>&1;
python tune.py beijing > log_beijing 2>&1;
python tune.py chongqing > log_chongqing 2>&1;
python tune.py fujian > log_fujian 2>&1;
python tune.py gansu > log_gansu 2>&1;
python tune.py guangdong > log_guangdong 2>&1;
python tune.py guangxi > log_guangxi 2>&1;
python tune.py guizhou > log_guizhou 2>&1;
python tune.py hainan > log_hainan 2>&1;
python tune.py hebei > log_hebei 2>&1;
python tune.py heilongjiang > log_heilongjiang 2>&1;
python tune.py henan > log_henan 2>&1;
python tune.py hubei > log_hubei 2>&1;
python tune.py hunan > log_hunan 2>&1;
python tune.py jiangsu > log_jiangsu 2>&1;
python tune.py jiangxi > log_jiangxi 2>&1;
python tune.py jilin > log_jilin 2>&1;
python tune.py liaoning > log_liaoning 2>&1;
python tune.py ningxia > log_ningxia 2>&1;
python tune.py qinghai > log_qinghai 2>&1;
python tune.py shaanxi > log_shaanxi 2>&1;
python tune.py shandong > log_shandong 2>&1;
python tune.py shanghai > log_shanghai 2>&1;
python tune.py shanxi > log_shanxi 2>&1;
python tune.py sichuan > log_sichuan 2>&1;
python tune.py tianjin > log_tianjin 2>&1;
python tune.py tibeta > log_tibet 2>&1;
python tune.py xinjiang > log_xinjiang 2>&1;
python tune.py yunnan > log_yunnan 2>&1;
python tune.py zhejiang > log_zhejiang 2>&1;
python tune.py inner\ mongolia > log_inner_mongolia 2>&1;

# # 激活 Conda 环境
# conda activate cs0

# # 确保 Conda 钩子被正确加载
# eval "$(conda shell.bash hook)"

# # 第一组
# python tune.py anhui > log_anhui 2>&1 &
# python tune.py beijing > log_beijing 2>&1 &
# wait

# # 第二组
# python tune.py chongqing > log_chongqing 2>&1 &
# python tune.py fujian > log_fujian 2>&1 &
# wait

# # 第三组
# python tune.py gansu > log_gansu 2>&1 &
# python tune.py guangdong > log_guangdong 2>&1 &
# wait

# # 第四组
# python tune.py guangxi > log_guangxi 2>&1 &
# python tune.py guizhou > log_guizhou 2>&1 &
# wait

# # 第五组
# python tune.py hainan > log_hainan 2>&1 &
# python tune.py hebei > log_hebei 2>&1 &
# wait

# # 第六组
# python tune.py heilongjiang > log_heilongjiang 2>&1 &
# python tune.py henan > log_henan 2>&1 &
# wait

# # 第七组
# python tune.py hubei > log_hubei 2>&1 &
# python tune.py hunan > log_hunan 2>&1 &
# wait

# # 第八组
# python tune.py jiangsu > log_jiangsu 2>&1 &
# python tune.py jiangxi > log_jiangxi 2>&1 &
# wait

# # 第九组
# python tune.py jilin > log_jilin 2>&1 &
# python tune.py liaoning > log_liaoning 2>&1 &
# wait

# # 第十组
# python tune.py ningxia > log_ningxia 2>&1 &
# python tune.py qinghai > log_qinghai 2>&1 &
# wait

# # 第十一组
# python tune.py shaanxi > log_shaanxi 2>&1 &
# python tune.py shandong > log_shandong 2>&1 &
# wait

# # 第十二组
# python tune.py shanghai > log_shanghai 2>&1 &
# python tune.py shanxi > log_shanxi 2>&1 &
# wait

# # 第十三组
# python tune.py sichuan > log_sichuan 2>&1 &
# python tune.py tianjin > log_tianjin 2>&1 &
# wait

# # 第十四组
# python tune.py tibeta > log_tibet 2>&1 &
# python tune.py xinjiang > log_xinjiang 2>&1 &
# wait

# # 第十五组
# python tune.py yunnan > log_yunnan 2>&1 &
# python tune.py zhejiang > log_zhejiang 2>&1 &
# wait

# # 第十六组
# python tune.py inner\ mongolia > log_inner_mongolia 2>&1 &
