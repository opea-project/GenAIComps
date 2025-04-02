# The core features for clip finetune tool:

Below method can run on Classification task and Image to Text task

<table width="100%">
    <tr>
        <td align="center" colspan="1"><strong>Method</strong></td>
        <td align="center" colspan="1"><strong>Detail Description</strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong>Full Finetune</strong></td>
        <td align="center" colspan="1"><strong>1. Default update all parameters<br>
        2. Enable <a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3155_ECCV_2020_paper.php">Angle-Based Selection</a>(base on the weight angle to determine which layer to update)<br>
        3. Enable <a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3155_ECCV_2020_paper.php">Angle-Based Selection</a>(base on the weight angle to determine which layer to update)<br></strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong>Partial Finetuning - bias</strong></td>
        <td align="center" colspan="1"><strong>1. Default update all bias parameters<br>
        2. Allow users to customize which layers participate in training and which ones do not<br></strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong><a href="https://arxiv.org/abs/2203.12119">Prompt Tuning</a></strong></td>
        <td align="center" colspan="1"><strong>adding prompt embedding layer at the head of model or at the inputs of every layer and only train these layers<br></strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong><a href="https://arxiv.org/pdf/2110.04544">Adapter Tuning</a></strong></td>
        <td align="center" colspan="1"><strong>adding adapter network at the end of encoder and only train this network<br></strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong>Training free - <a href="https://arxiv.org/pdf/2207.09519">Tip Adapter</a></strong></td>
        <td align="center" colspan="1"><strong>1. finetune CLIP model without any training or with few epochs learning<br>
        2. Added fixed cache size to reduce memory and enable experience sharing across different datasets</strong></td>       
    <tr>
</table>
