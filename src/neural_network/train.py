from PyQt5.QtCore import QThread, pyqtSignal
import torch

from nets.ssd import get_ssd
from nets.ssd_training import Generator,MultiBoxLoss
from utils.config import Config
import torch.backends.cudnn as cudnn
# import torch
import numpy as np
import torch.optim as optim


class train(QThread):
    msg = pyqtSignal(str)
    finish= pyqtSignal(str)


    def __init__(self):
        QThread.__init__(self)
        self.flag=True


    def adjust_learning_rate(self,optimizer, lr, gamma, step):
        lr = lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def run(self):
        Batch_size = Config["Batch_size"]  # 每批次输入图片数量
        lr = Config["lr"]
        Epoch = Config["Epoch"]
        Cuda = Config["Cuda"]
        Start_iter = Config["Start_iter"]
        model = get_ssd("train", Config["num_classes"])

        self.msg.emit('将权重加载到字典中......')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(Config['migrate_path'], map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.msg.emit('完成加载')

        net = model
        if Cuda:
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            net = net.cuda()

        annotation_path = 'neural_network/2007_train.txt'
        with open(annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)  # 打乱
        np.random.seed(None)
        num_train = len(lines)

        # 生成图片和标签
        gen = Generator(Batch_size, lines,
                        (Config["min_dim"], Config["min_dim"]), Config["num_classes"]).generate()

        # 优化器
        optimizer = optim.Adam(net.parameters(), lr=lr)
        criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, Cuda)

        net.train()

        epoch_size = num_train // Batch_size
        for epoch in range(Start_iter, Epoch):
            if epoch % 10 == 0:
                self.adjust_learning_rate(optimizer,lr, 0.95, epoch)
            self.loc_loss = 0
            self.conf_loss = 0
            for iteration in range(epoch_size):
                if self.flag==True:
                    images, targets = next(gen)
                    with torch.no_grad():
                        if Cuda:
                            images = torch.from_numpy(images).cuda().type(torch.FloatTensor)
                            targets = [torch.from_numpy(ann).cuda().type(torch.FloatTensor) for ann in targets]
                        else:
                            images = torch.from_numpy(images).type(torch.FloatTensor)
                            targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                    # 前向传播
                    out = net(images)
                    # 清零梯度
                    optimizer.zero_grad()
                    # 计算loss
                    loss_l, loss_c = criterion(out, targets)
                    loss = loss_l + loss_c
                    # 反向传播
                    loss.backward()
                    optimizer.step()

                    # 加上
                    self.loc_loss += loss_l.item()
                    self.conf_loss += loss_c.item()

                    # print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
                    self.msg.emit('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
                    loss = '|| Loc_Loss: %.4f || Conf_Loss: %.4f ||' % (
                    self.loc_loss / (iteration + 1), self.conf_loss / (iteration + 1))
                    self.msg.emit('iter:' + str(iteration) + '/' + str(epoch_size) + str(loss))
                else:
                    break
            else:
                if self.loc_loss /(iteration + 1)<=Config["loc_loss"] and self.conf_loss / (iteration + 1)<=Config["conf_loss"]:
                    ep = 'neural_network/outputs/Epoch%d-Loc%.4f-Conf%.4f.pth' % (epoch + 1), self.loc_loss / (
                                iteration + 1), self.conf_loss / (iteration + 1)
                    self.msg.emit('保存模型' + str(ep))
                    torch.save(model.state_dict(), 'neural_network/outputs/Epoch%d-Loc%.4f-Conf%.4f.pth' % (
                        (epoch + 1), self.loc_loss / (iteration + 1), self.conf_loss / (iteration + 1)))
                continue
            self.msg.emit('训练中止')
            break
        else:
            ep='neural_network/outputs/Epoch%d-Loc%.4f-Conf%.4f.pth' % ((epoch + 1), self.loc_loss / (iteration + 1), self.conf_loss / (iteration + 1))
            self.msg.emit('保存最后模型' + str(ep))
            torch.save(model.state_dict(), 'neural_network/outputs/Epoch%d-Loc%.4f-Conf%.4f.pth' % (
                (epoch + 1), self.loc_loss / (iteration + 1), self.conf_loss / (iteration + 1)))
            self.finish.emit('训练完成')



