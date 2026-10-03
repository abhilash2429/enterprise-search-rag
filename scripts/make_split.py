from entsearch.data import SPLITS, load_questions
from entsearch.split import stratified_split

q = load_questions()
dev, test = stratified_split(q)
SPLITS.mkdir(parents=True, exist_ok=True)
(SPLITS / "dev.txt").write_text("\n".join(dev) + "\n")
(SPLITS / "test.txt").write_text("\n".join(test) + "\n")

by_type = q.assign(split=q.question_id.isin(set(dev)).map({True: "dev", False: "test"}))
print(by_type.groupby(["question_type", "split"]).size().unstack().to_string())
print(f"dev {len(dev)}  test {len(test)}")
