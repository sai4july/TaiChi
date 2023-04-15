import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer,AutoConfig,BertTokenizer,BertModel,BertForSequenceClassification
import textattack.models.wrappers as wp
from textattack.models.wrappers import ModelWrapper,PyTorchModelWrapper

class CustomModelWrapper(PyTorchModelWrapper):
    def __init__(self,model,tokenizer):
        super(CustomModelWrapper,self).__init__(model,tokenizer)

    def __call__(self,text_input_list):
        inputs_dict = self.tokenizer(
            text_input_list,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        return outputs['logits']


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
        
class Backbone(nn.Module):
    def __init__(self, num_labels, BERT_MODEL_NAME, type = "clean", freeze_bert=False):
        super().__init__()
        self.type = type
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(BERT_MODEL_NAME, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size,num_labels)  
        self._init_weights(self.fc)

        if freeze_bert:
            print("freezing bert parameters")
            for param in self.bert.parameters():
                param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def feature(self,ids,mask):
        outputs = self.bert(input_ids=ids,attention_mask=mask)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states,mask)
        return feature

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        feature = self.feature(input_ids,attention_mask)
        logits  = self.fc(feature)
        loss = 0
        if labels is not None:

            return {'loss':loss,"logits":logits}
        else:
            return {"logits":logits}



model_path = ""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
model = Backbone(6,"bert-base-uncased")
model.load_state_dict(torch.load(model_path))
model = CustomModelWrapper(model,tokenizer)


