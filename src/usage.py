import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import typer
from safetensors.torch import load_file
from transformers import AutoTokenizer


from src.dataset import pad_tensors
from src.model import RetrievalModel
from src.configurations.validation import load_config

app = typer.Typer()
config = load_config()
PROJ_ROOT = Path(__file__).resolve().parents[1]
SAVE_MODEL_PATH = PROJ_ROOT / "models"


@app.command()
def main():
    model = RetrievalModel().eval()
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        model.load_state_dict(load_file(SAVE_MODEL_PATH / 'retrieval-200.safetensors'))

        query = 'query: Where when and how did the idea of “reasonableness” originate?'
        query = tokenizer.encode_plus(query).encodings[0]
        documents = [
            "passage: See Harold J. Berman, &quot;The Origins of Historical Jurisprudence: Coke, Selden, Hale&quot; (1994) 103 Yale Law Journal 1651, p. 1691, n. 101:</p>\n<blockquote>\n<p>the translation of &quot;reason&quot; into &quot;reasonableness&quot; and the exaltation of &quot;common sense&quot; are English developments of the seventeenth century, to which Coke contributed.</p>\n</blockquote>\n<p>At p. 1718-19:</p>\n<blockquote>\n<p>Coke had said it is the nature of law to be reasonable, and that the test of reasonableness is its ability to withstand the test of time.</p>\n</blockquote>\n<p>See also S.E. Thorne, &quot;Dr. Bonham(1938) 54 Law Quarterly Review 543, p. 543:</p>\n<blockquote>\n<p>To students of the origins of American constitutional law and theory no judicial utterance of Sir Edward Coke can surpass in interest and importance his so-called <em>dictum</em> in Dr. Bonhams case</a>, decided in the Court of Common Pleas in 1610. It is widely regarded as foreshadowing not merely the power which American courts to-day exercise in the disallowance of statutes on the ground of their conflict with the constitution, but also that very test of reasonableness which is the ultimate flowing of that power.</p>\n</blockquote>\n<p>This concept as a ground of review arose in the context of &quot;conflict between Parliament and the Crown over the nature and limits of prerogative and the common law&quot; (Michael Foran, &quot;The Constitutional Foundations of Reasonableness Review: Artificial Reason and Wrongful Discrimination&quot; (2022) 26:3 Edinburgh Law Review 295, p. 299).</p>\n<p>In one case, the, Coke repealed a judgment of King James I, on the basis that the judgment was not grounded in the common law. Coke said: &quot;causes which concern the life, or inheritance, or goods, or fortunes of his subjects, are not to be decided by natural reason but by the <em>artificial reason and judgment of law</em>.&quot;</p>\n<p>This is in contrast to a view that Crown action would not be substantively reviewable. By defining reasonableness as something that can only be determined through the wisdom of judges, Coke was broadening the judicial power.",
            "passage: <p>The  &quot;reasonable man&quot; standard in the common law of torts</a> is sometimes attributed to the English case of Vaughan v. Menlove</a></em> (1837).</p>\n",
            "passage: <p>If you pushed her back <strong>after</strong> she slapped you <strong>and</strong> it is not clear that a second slapping would occur (<strong>or</strong> she slapped you, because you pushed her)</p>\n<ul>\n<li>then it is <strong>not</strong> self-defence\n<ul>\n<li>you (<strong>or</strong> she) did not <strong>prevent a present</strong> unlawful attack</li>\n</ul>\n</li>\n</ul>\n<p>If it is clear that you are <strong>going to be</strong> slapped</p>\n<ul>\n<li>then pushing her away, in a reasonable manor, is self-defence\n<ul>\n<li>you <strong>prevented a present</strong> unlawful attack</li>\n</ul>\n</li>\n</ul>\n<p>What is considered <strong>reasonable</strong> will later be determined by a judge.</p>\n<hr />\n<blockquote>\n\§ 32 - Self-defence StGB</a><br />\n(2) ‘Self-defence’ means any defensive action which is necessary <strong>to avert a present</strong> unlawful attack on oneself or another.</p>\n</blockquote>\n<hr />\n<p><strong>Sources</strong>:</p>\n<ul>\n<li>§ 32 - Self-defence StGB</a></li>\n</ul>\n"
        ]
        documents = tokenizer(documents, max_length=512, truncation="longest_first").encodings
        document_input_ids = pad_tensors([torch.tensor(x.ids) for x in documents])
        document_attention_mask = pad_tensors([torch.tensor(x.attention_mask) for x in documents])

        query_vec = model(torch.tensor(query.ids)[None, :], torch.tensor(query.attention_mask)[None, :], is_query=True)
        document_vec = model(document_input_ids, document_attention_mask, is_query=False)

        cosine_similarities = query_vec @ document_vec.T

        print(cosine_similarities)


if __name__ == '__main__':
    app()