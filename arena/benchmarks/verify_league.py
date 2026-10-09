"""Re-execute every recorded action and validate every tick, without loading AIs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time

from generals_arena.league.game import replay_frames
from generals_arena.league.store import LeagueStore


def verify(item):
    directory,game_id=item
    store=LeagueStore(directory)
    result,blob=store.replay(game_id)
    replay_frames(blob,result,stride=result['ticks'],compact=True)
    return result['ticks']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    store=LeagueStore(args.directory)
    with store.connect() as db:
        ids=[row[0] for row in db.execute("SELECT id FROM games WHERE status='complete' ORDER BY sequence")]
    start=time.perf_counter()
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        ticks=sum(pool.map(verify,((args.directory,key) for key in ids),chunksize=4))
    print(json.dumps({'verified_games':len(ids),'verified_ticks':ticks,'seconds':time.perf_counter()-start},indent=2))


if __name__=='__main__':main()
