import argparse


# for training
parser = argparse.ArgumentParser()
# parser.add_argument("--train",default=True,help="learning rate",action="store_true")
# parser.add_argument("--predict",default=False,help="learning rate",action="store_true")
parser.add_argument("--lr",default=1e-3,help="learning rate",type = float)
parser.add_argument("-b","--batch_size",default=16,help="batch_size",type = int)
parser.add_argument("-e","--epoch",default=300,help="num_epoch",type = int)
parser.add_argument("-worker","--num_worker",default=4,help="num_worker",type = int)
parser.add_argument("-class","--num_class",default=1,help="num_class",type = int)
parser.add_argument("-c","--in_channels",default=1,help="in_channels",type = int)
parser.add_argument("-size","--image_size",default=224,help="image_size",type = int)
parser.add_argument("-flow","--train_flow",default=5,help="image_size",type = int)
parser.add_argument("-data_path","--data_path",default="/home/student/Desktop/SegmentationThesis/data/microglia/",help="data path",type = str)
parser.add_argument("-train_txt","--train_txt",default="/home/student/Desktop/SegmentationThesis/data/train.txt",help="train_txt",type = str)
parser.add_argument("-val_txt","--val_txt",default="/home/student/Desktop/SegmentationThesis/data/validation.txt",help="val_txt",type = str)
parser.add_argument("-test_txt","--test_txt",default="/home/student/Desktop/SegmentationThesis/data/test.txt",help="test_txt",type = str)
parser.add_argument("-unlabel_txt","--unlabel_txt",default="/home/student/Desktop/SegmentationThesis/GAN/FakeImage/FakeAll.txt",help="unlabel_txt",type = str)
parser.add_argument("-consistency_weight","--consistency_weight",default=1,help="consistency_weight",type = float)
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')


args = parser.parse_args()
