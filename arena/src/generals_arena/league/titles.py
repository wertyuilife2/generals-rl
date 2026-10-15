"""Presentation-only 1v1 titles; never affect ratings or matchmaking.

Codeforces title names / bands:
https://codeforces.com/blog/entry/20638
https://codeforces.com/blog/entry/59228 (Master 2100, LGM 3000)
Arena's GoR score is not a Codeforces performance equivalence.
"""
from .rating import PROVISIONAL_GAMES

_BANDS = (
    (None, 'newbie', 'Newbie', '#aaaaaa'),
    (1200, 'pupil', 'Pupil', '#82bf91'),
    (1400, 'specialist', 'Specialist', '#67c7c7'),
    (1600, 'expert', 'Expert', '#7ba9ed'),
    (1900, 'candidate-master', 'Candidate Master', '#c29af0'),
    (2100, 'master', 'Master', '#edbd70'),
    (2300, 'international-master', 'International Master', '#e9a45b'),
    (2400, 'grandmaster', 'Grandmaster', '#ee8585'),
    (2600, 'international-grandmaster', 'International Grandmaster', '#ed6666'),
    (3000, 'legendary-grandmaster', 'Legendary Grandmaster', '#ff5252'),
)


def title_bands():
    return [dict(minimum=minimum, maximum=_BANDS[i+1][0] if i+1 < len(_BANDS) else None,
                 key=key, label=label, color=color)
            for i,(minimum,key,label,color) in enumerate(_BANDS)]


def rating_title(elo, rated_games):
    if rated_games < PROVISIONAL_GAMES:
        return dict(key='unrated',label='Unrated',color='#aaaaaa',
                    minimum_games=PROVISIONAL_GAMES,remaining_games=PROVISIONAL_GAMES-rated_games)
    return next(band for band in reversed(title_bands()) if band['minimum'] is None or elo >= band['minimum'])
