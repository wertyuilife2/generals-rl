#ifndef AEGIS_SOURCE
#define AEGIS_SOURCE "../policy.cpp"
#endif
#include AEGIS_SOURCE
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>

// Exercise the actual decision entry point through a quiet, fogged opening.
// Waiting is useful because one accumulated column can open a long road before
// the first land-growth pulse; early single-soldier dispatches spend commands
// traversing that same road repeatedly.
struct OpeningBoard {
    int side, seat, home;
    std::vector<std::uint8_t> terrain, structure, visible, fog;
    std::vector<std::int16_t> owner;
    std::vector<std::int64_t> army;
    std::array<std::uint8_t, 2> alive{1, 1}, flags{};
    std::array<std::int64_t, 2> total{1, 1}, land{1, 1}, cities{};
    std::array<double, 2> departure{-1, -1};
    ArenaObservation view;

    OpeningBoard(int width, int player)
        : side(width), seat(player), home(width * (width / 2) + width / 2),
          terrain(width * width, 255), structure(width * width, 255),
          visible(width * width), fog(width * width), owner(width * width, -3),
          army(width * width, -1),
          view{width, width, 2, player, 0, home, terrain.data(), structure.data(),
               owner.data(), army.data(), visible.data(), alive.data(), total.data(),
               land.data(), cities.data(), fog.data(), flags.data(), flags.data(),
               departure.data(), nullptr} {
        for (int row = -1; row <= 1; ++row)
            for (int col = -1; col <= 1; ++col) {
                int cell = home + row * side + col;
                terrain[cell] = structure[cell] = 0;
                visible[cell] = 1;
                owner[cell] = -1;
                army[cell] = 0;
            }
        owner[home] = seat;
        structure[home] = 2;
        army[home] = 1;
    }

    void advance_to(int tick) {
        view.tick = tick;
        army[home] = 1 + tick / 2;
        total.fill(army[home]);
    }
};

int main(int argc, char **argv) {
    std::string params = "{}";
    if (argc > 1) {
        std::ifstream input(argv[1]);
        assert(input.good());
        params.assign(std::istreambuf_iterator<char>(input), {});
    }
    for (int side : {15, 25, 35}) for (int seat : {0, 1}) {
        OpeningBoard board(side, seat);
        auto agent = std::make_unique<aegis::Agent>(seat, 0, params.c_str());
        for (int tick = 0; tick < 24; ++tick) {
            board.advance_to(tick);
            assert(agent->act(&board.view) == 0);
        }
        board.advance_to(24);
        auto action = agent->act(&board.view);
        assert(action > 0);
        assert((action - 1) / 8 == board.home);
        assert((action - 1) % 2 == 0); // Full accumulated column, leaving one.
        assert(board.army[board.home] == 13);

        // A visible capturable enemy general interrupts opening accumulation.
        OpeningBoard tactical(side, seat);
        tactical.advance_to(10);
        tactical.owner[tactical.home + 1] = 1 - seat;
        tactical.structure[tactical.home + 1] = 2;
        tactical.army[tactical.home + 1] = 2;
        tactical.total[1 - seat] = 2;
        auto emergency = std::make_unique<aegis::Agent>(seat, 0, params.c_str());
        assert(emergency->act(&tactical.view) == arena_move(tactical.home, 1));
    }
    std::cout << "12 opening scenarios passed: wait through tick 23, dispatch at 24, tactical override\n";
}
