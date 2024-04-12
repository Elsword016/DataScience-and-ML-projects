## Segmentation of dendritic spines using Segment Anything and prompting with points

Segment Anything (SAM) is truly a remarkable foundation model for image segmentation, with its unique 'promptable segmentation' - ROIs can be defined by bounding boxes, points and even text (weights not released). It is one of my favorite models to come out last year.
Being a biologist, segmentation is used quite frequently in my domain, and having this kind of promptable segmentation helps a lot. 

From its inception, people have been finetuning SAM for various use cases but most of the time it's done (so far I came across) using bounding boxes as prompts, however many cases drawing a bounding box on a object of interest might not be possible or difficult. So I tried
to finetune the base model using points as prompts rather than bounding boxes. 

That being said, I think there could be much better ways to do it, however, I kept it simple for now.

**Segment Anything** - [[Repo]](https://github.com/facebookresearch/segment-anything) [[Paper]](https://ai.meta.com/research/publications/segment-anything/)

**How to finetune SAM by Encord** - [[Webpage]](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)

**Awesome Segment Anything- Collection of papers and repositories for SAM** - [[Repo]](https://github.com/Hedlen/awesome-segment-anything)

![image](https://github.com/Elsword016/DataScience_portfolio/assets/29883365/11c83053-a8d4-41f7-905e-deaec2ccf127)


