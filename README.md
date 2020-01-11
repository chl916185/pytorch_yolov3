使用yolov3模型进行目标检测测试。参考网址：https://blog.paperspace.com/tag/series-yolo/   
1、没有训练模型，直接使用yolov3.cfg和yolov3.weights初始化网络，然后直接进行测试。yolov3.weights自行下载。   
2、解析cfg文件时，一定要注意空格行、注释行以及[]、空格的处理。   
3、构建卷积层时，要注意以下几点：   
    （1）batchnorm为1时(存在时)，即存在bn层，同时注意卷积bias不存在；若batchnorm不存在时，即不存在bn层，但卷积bias存在。   
    （2）pad为1时，padding要根据kernel_size设置；pad为0时，padding为0。   
4、构建shortcut层和route层时，都使用一个空层。在forward时，前者是add，后者是cat。   
5、构建yolo层时，使用一个空层，且记录当前使用的anchors。anchors是两个一对(w,h)。   
6、forward中yolo层输出的特征图(13 * 13,26 * 26,52 * 52)，皆要进行数据转换：   
    (1)根据输入原图大小与特征图大小确定stride，grid_size。同时要对anchors进行放缩。   
    (2)对特征图进行view,[batch_size,grid_size * grid_size * num_anchors,num_classes+5]。   
    (3)sigmoid使用。对中心点x,y的预测以及对类别的预测(多标签分类)。   
    (4)根据公式更新center_x,center_y,w,h。注意对引入的新的tensor转换device(cpu or gpu)。   
    (5)最后要把预测的前四项(代表位置信息)转移到原图上，即用stride进行放缩。   
    (6)三个不同大小的特征图都要进行此种转换，并在dim=1上进行cat。   
7、使用weightfile读取权重时，要注意数据类型，数据分布顺序，conv是否存在bias，bn是否存在。   
8、网络的输入dim最好是32的整数倍，且大于32，一般取416。同时要注意对数据的预处理：   
    (1)输入图像的尺寸不一定符合要求，所以要resize成规定的输入dim，同时要注意纵横比要保持不变(位置信息很重要)。   
    (2)输出结果的位置信息数据是符合输入dim的，所以要转换到原图尺寸上。   
9、对特征图进行view后，num_classes+5要进行数据转换，后面的class score取最高分数和最高分数对应的索引，   
    即变成2+5。同时最后要加入图片索引信息。   
10、特征图view时,一定要注意结构和存储的影响。transpose之后往往要进行深拷贝(copy 或 contiguous)。   
    prediction = prediction.view(batch_size, num_property * num_anchors, grid_size * grid_size) #b,c,h,w->b,c,h * w  
    prediction = prediction.transpose(1, 2).contiguous()   #b,c,h * w->b,h * w,c  
    prediction=prediction.view(batch_size,grid_size * grid_size * num_anchors,num_property) #b,h * w * a,p (c=a * p)    
11、测试：python detect.py --images dog-cycle-car.png --det det  
