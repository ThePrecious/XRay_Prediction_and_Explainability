import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

class ChestXrayEnsemble(pl.LightningModule):
    def __init__(self,num_classes,base_learners):
        super(ChestXrayEnsemble,self).__init__()

        self.num_classes = num_classes
        self.n_learners = len(base_learners)
        self.base_learners = nn.ModuleList([self.get_pretrained_basemodels(learner) for learner in base_learners])
        self.linears = nn.ModuleList([nn.Linear(4*self.n_learners * self.n_learners,self.n_learners) for i in range(self.num_classes)])

    def get_pretrained_basemodels(self,model_name):
        model = xrv.models.get_model(model_name)
        for param in model.named_parameters():
            param[1].requires_grad = False

        return model

    def forward_class_weights(self,class_preds,learner):
        b = torch.flatten(class_preds,start_dim=0)
        c = torch.outer(b,b)
        r = int(c.shape[0]/(2*self.n_learners))

        w_features = None
        for i in range(r):
            r_frm = i*2*self.n_learners
            r_to = (i+1) * 2 * self.n_learners
            w = torch.flatten(c[r_frm:r_to,r_frm:r_to],start_dim=0)
            w = w.unsqueeze(0)

            if w_features is None:
                w_features = w
            else:
                w_features = torch.cat([w_features,w])

        x = learner(w_features)
        weights = torch.nn.functional.softmax(x,dim=1).unsqueeze(1)
        return weights

    def forward(self,x):

        preds_all = None
        for i in range(self.n_learners):
            if i == 0:
                preds_all = self.base_learners[i](x)[:,:self.num_classes].unsqueeze(1)
            else:
                preds_all = torch.cat([preds_all,
                                       self.base_learners[i](x)[:,:self.num_classes].unsqueeze(1)
                                      ],dim=1)

        preds = torch.cat([preds_all,1-preds_all],dim=1)

        class_weights = None
        for i in range(self.num_classes):
            cw = self.forward_class_weights(preds[:,:,i],self.linears[i])

            if class_weights is None:
                class_weights = cw
            else:
                class_weights = torch.cat([class_weights,cw],dim=1)

        class_weights = torch.permute(class_weights,(0,2,1))

        output1 = torch.multiply(class_weights,preds_all)
        output = torch.sum(output1,dim=1)

        return output
