#include <iostream>

#include "mtree.hpp"

using namespace std;

class Pair {
public:
    Pair() : empty(true), x(0), y(0) {}
    Pair(int x, int y) : empty(false), x(x), y(y) {}

    bool is_empty() {
        return empty;
    }

    static Pair* random() {
        random_device rd;
        return new Pair(rd() % 100, rd() % 100);
    }

    int x, y;

private:
    bool empty;
};

class ManhattanDistance : public IMetric<Pair> {
public:
    int distance(Pair &p1, Pair &p2) {
        return abs(p1.x - p2.x) + abs(p1.y - p2.y);
    }
};

int main() {
    ManhattanDistance metric;
    MTree<Pair, RandomSplit<Pair, 10>, 10> mt (&metric);

    cout << "Inserting...";
    for (int i = 0; i < 10000; i++) {
        mt.insert(*Pair::random());
    }
    cout << " done." << endl;
    
    vector<Pair> ret;
    Pair pt (0, 0);
    mt.range(pt, 3, ret);

    for (Pair p : ret) {
        printf("%d, %d\n", p.x, p.y);
    }
}
