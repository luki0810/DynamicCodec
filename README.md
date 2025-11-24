# DynamicCodec
Implementing dynamic composition of Codec using Argbind and ClassChoice


## 通用运行指令

```cmd
# ./run.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
python main.py \
--load_path conf/base.yaml \
--save_path runs/test/args.yaml \
--args.debug 1
```



## 其他运行指令

查看./bash.sh内的内容



## 加入新的组件

需要完成一系列步骤：

- model/decoder、model/encoder、model/quantizer内加入模型代码，需要修改为继承自AbsEncoder/AbsDecoder/AbsQuantizer
- model/all_choices.py中加入组件选项
- conf/model/encoder、 conf/model/decoder、conf/model/quantizer中配置组件参数



## 运行新的组件

只需要修改conf/base.yaml内的选择

