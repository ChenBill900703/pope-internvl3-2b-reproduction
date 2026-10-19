"""Regenerate figures solely from published aggregates, without model/data access."""
from build_release import read,charts
if __name__=='__main__':charts(read('paper_reference.json'))
