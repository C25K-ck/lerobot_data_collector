import yaml
 
# 打开并读取 YAML 文件
with open('/home/lifeng/code/act_eval/conf.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
 
# 打印读取的内容
print(config)
 
# 访问具体值
print("Database Host:", config['get_motor']['model'])

