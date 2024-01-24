from torch import nn
#from PositionalEncoding import PositionalEncoding1d

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

class PairNet(nn.Module):
    def __init__(self, dim):

        super().__init__()

        #self.emb = nn.Embedding(3, 128)

        #self.pos_encoding = PositionalEncoding1d(128, dim)


        #-----------------------------------------------
        self.reduction = nn.Sequential(
            Reshape(-1,1,dim),
            nn.Linear(dim, 4096),
            nn.ReLU(),
        )
        
        #--------------------------------------------
        
        self.conv = nn.Sequential(

            nn.Conv1d(1, 256, 2, 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            #----------------------------------------

            nn.Conv1d(256, 256, 2, 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 256, 2, 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 256, 2, 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            #----------------------------------

            nn.Conv1d(256, 512, 2, 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            #----------------------------------            

            nn.Conv1d(512, 512, 2, 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 512, 2, 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(512, 512, 2, 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            #----------------------------------

            nn.Conv1d(512, 1024, 2, 2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            #----------------------------------

            nn.AdaptiveAvgPool1d(1)
        )



        #-------------------------------------
        self.aggregate = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        #x = x.long()
        #x = self.emb(x); print("ebm", x.shape)
        #x = x + self.pos_encoding(x); print("enc", x.shape)
        #x = x.transpose(-1, -2)
        x = self.reduction(x)
        x = self.conv(x)
        x = x.view(-1, 1024)
        out = self.aggregate(x)
        return out


class PairNetClassifier(nn.Module):
    """model of pairnet
    """
    def __init__(self, dim):
        super().__init__()
        self.pairnet = PairNet(dim)
        self.classifier = nn.Sigmoid()

    def forward(self, x):
        x = self.pairnet(x)
        out = self.classifier(x)
        return out
