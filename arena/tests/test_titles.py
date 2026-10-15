from generals_arena.league.titles import rating_title, title_bands


def test_titles_use_numeric_boundaries_and_keep_unplayed_players_unrated():
    bands=title_bands()
    assert rating_title(4000,0)['key']=='unrated'
    assert rating_title(4000,5)['remaining_games']==1
    assert rating_title(-500,6)['key']=='newbie'
    for i,band in enumerate(bands[1:],1):
        assert rating_title(band['minimum'],6)==band
        assert rating_title(band['minimum']-.001,100)==bands[i-1]
    assert rating_title(1864.5,500)['key']=='expert'
    assert rating_title(2400,500)['key']=='grandmaster'
    assert rating_title(3000,500)['key']=='legendary-grandmaster'
