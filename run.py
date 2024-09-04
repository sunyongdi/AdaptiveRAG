
from pprint import pprint
from Graph import app

# Run
# inputs = {"question": "《甘肃祁连山国家级自然保护区管理条例2017》规定在自然保护区禁止哪些行为？"}
prompt = """你是一个法律审核专家，请根据审核规则对上下位法进行审查，判断下位法的规定是否与上位法存在冲突，返回yes 或 no；
            以JSON格式提供二进制分数，其中只有一个键“score”，没有前缀或解释。

            审核规则：
            下位法的立法目的是上位法所禁止的，构成冲突。

            下位法内容：
            司法部有关业务厅局、直属单位按照《全国专业标准化技术委员会管理办法》申请筹建全国专业标准化技术委员会，应当经司法部网信办研究论证后报批。
            
"""

inputs = {"question": prompt}
try:
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")
    
    # Final generation
    pprint(value["generation"])
except Exception as e:
    print('没有找到合适的内容')

