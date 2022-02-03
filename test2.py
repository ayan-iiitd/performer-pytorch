#%%
import tqdm
import torch
import torch.optim as optim
from performer_pytorch import PerformerEncDec
from torch.cuda.amp import autocast, GradScaler



#%%
NUM_BATCHES = int(1e5)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 16 + 2
ENC_SEQ_LEN = 32
DEC_SEQ_LEN = 64 + 1



#%%
# helpers

def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1]).bool().cuda()
        yield (src, tgt, src_mask, tgt_mask)



#%%
# instantiate model

model = PerformerEncDec(
    dim=512,
    enc_num_tokens=NUM_TOKENS,
    enc_depth=1,
    enc_heads=8,
    enc_max_seq_len=ENC_SEQ_LEN,
    enc_reversible=True,
    enc_feature_redraw_interval=1000,
    enc_nb_features = 64,
    dec_num_tokens=NUM_TOKENS,
    dec_depth=3,
    dec_heads=8,
    dec_max_seq_len=DEC_SEQ_LEN,
    dec_reversible=True,
    dec_feature_redraw_interval=1000,
    dec_nb_features=64
).cuda()



#%%
# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()



#%%


for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):

    ##### My lines start here

    src, tgt, src_mask, tgt_mask = next(cycle())





    break
    ##### My lines end here



    # model.train()

    # src, tgt, src_mask, tgt_mask = next(cycle())

    # with autocast():
    #     loss = model(src, tgt, enc_mask=src_mask, dec_mask=tgt_mask)

    # scaler.scale(loss).backward()
    # print(f'{i}: {loss.item()}')

    # scaler.step(optim)
    # scaler.update()
    # optim.zero_grad()

    # if i != 0 and i % GENERATE_EVERY == 0:
    #     model.eval()
    #     src, _, src_mask, _ = next(cycle())
    #     src, src_mask = src[:1], src_mask[:1]
    #     start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

    #     sample = model.generate(src, start_tokens, ENC_SEQ_LEN, enc_mask=src_mask)
    #     incorrects = (src != sample).abs().sum()

    #     print(f"input:  ", src)
    #     print(f"predicted output:  ", sample)
    #     print(f"incorrects: {incorrects}")



#%%
test_sample = {"src_txt": ["turkey has blocked access to twitter and youtube after they refused a request to remove pictures of a prosecutor held during an armed siege last week .", "a turkish court imposed the blocks because images of the deadly siege were being shared on social media and ` deeply upset ' the wife and children of mehmet selim kiraz , the hostage who was killed .", "the 46-year-old turkish prosecutor died in hospital when members of the revolutionary people 's liberation party-front ( dhkp-c ) stormed a courthouse and took him hostage .", "the dhkp-c is considered a terrorist group by turkey , the european union and us .", "a turkish court has blocked access to twitter and youtube after they refused a request to remove pictures of prosecutor mehmet selim kiraz held during an armed siege last week", "grief : the family of mehmet selim kiraz grieve over his coffin during his funeral at eyup sultan mosque in istanbul , turkey .", "he died in hospital after he was taken hostage by the far-left organisation", "two of his captors were killed when security forces took back the building where the far-left group was holding him .", "gunshots were heard and smoke could be seen rising from the scene at the end of the six-hour stand-off .", "mr kiraz , a father-of-two married to a judge who also worked at the courthouse , was targeted for his part in an investigation into the death of berkin elvan .", "the 15-year-old was severely wounded after being hit on the head by a tear-gas canister fired by a police officer during anti-government protests in istanbul in june 2013 .", "after spending 269 days in a coma , elvan eventually died on march 11 last year .", "his death , and the subsequent investigation , have since become a rallying point for the country 's far-left .", "gathering : prosecutors , lawyers and judges stand near a statue of lady justice during the funeral ceremony", "a british national , of polish origin but who has not been named , was arrested on saturday as part of an operation against the revolutionary people 's liberation party-front , according to reports .", "a foreign office spokeswoman said this morning : ' i can confirm that a british national has been arrested in turkey and that we are offering consular assistance . '", "before imposing the blocks on the websites , turkish authorities had tried to prevent newspapers printing images taken during the siege last week .", "the newspapers were accused by the government of ` spreading terrorist propaganda ' in sharing the images of the hostage-taking .", "presidential spokesman ibrahim kalin said : ` this has to do with the publishing of the prosecutor 's", "what happened in the aftermath ( of the prosecutor 's", "killing ) is as grim as the incident itself .", "` the demand from the prosecutor 's office is that this image", "not be used anywhere in electronic platforms .", "` the wife and children of prosecutor kiraz have been deeply", "the images are everywhere . '", "he added : ' a request has been made to both twitter and youtube for the", "removal of the images and posts but they have not accepted it", "and no response has been given .", "this decision has been taken through a court in istanbul . '", "critical : prosecutor mehmet selim kiraz was taken to hospital with gunshot wounds but died of his injuries", "strength of feeling : elvan has since become an icon for the turkish far-left and his supporters accuse the authorities of covering up the circumstances and perpetrators of his death", "google said it was working to restore service to the youtube", "video-sharing site , which it owns .", "working to restore access for its users .", "facebook said it had complied with a turkish court order requiring it to restrict access to some content or face a block on its service .", "a company spokesman said it would appeal the order .", "turkey 's telecoms regulator could not immediately be reached", "and there was no statement on its website .", "this is not the first time that turkish authorities have imposed blocks on social media sites and networks .", "in the run-up to local elections in march 2014 blocks were imposed after recordings circulated allegedly revealing corruption among senior officials .", "figures provided by twitter revealed that turkey filed more requests to remove content from the social network than any other nation between july and december 2014 ."], "tgt_txt": "turkish court imposed blocks as images of siege shared on social media<q>images ` deeply upset ' wife and children of hostage mehmet selim kiraz<q>prosecutor , 46 , died in hospital after hostages stormed a courthouse<q>two of his captors were killed when security forces took back the building"}



#%%
import torch
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer

#%%
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
t = tokenizer("Hello world")['input_ids']
r = tokenizer(" Hello world")['input_ids']


#%%
print(t)
print(r)


#%%
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
s = ' '.join(['Hello world! '] * 1000)
t2 = tokenizer(s)['input_ids']
print(t2)


#%%
from transformers import LongformerTokenizer
tokenizer2 = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
s = ' '.join(['Hello world! '] * 1000)
t4 = tokenizer2(s)['input_ids']
print(t4)


# %%
s = ' '.join(test_sample["src_txt"])
t4 = tokenizer2(s)['input_ids']
print(t4)


# %%
x = tokenizer2(s)


# %%
