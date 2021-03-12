import cbmos.cbmmodel._eventqueue as eventqueue
import cbmos.cell as cl

def test_constructor():
    cells = [cl.Cell(i, [0, 0, i]) for i in range(5)]
    for i, cell in enumerate(cells):
        cell.division_time = cell.ID

    event_list = [(i, cells[i]) for i in reversed(range(5))]

    queue = eventqueue.EventQueue(event_list)

    assert len(queue._events) == 5

    for i in range(5):
        t, cell= queue.pop()
        assert t == i
        assert len(cell) == 1
        assert cell[0].ID == i

def test_push():
    cell = cl.Cell(0, [0, 0])

    queue = eventqueue.EventQueue([])
    queue.push(1, cell)

    assert len(queue._events) == 1

    assert (1, cell) == queue._events[0]

    cell2 = cl.Cell(1, [0.25, 0.25])
    queue.push(2, cell2)

    assert len(queue._events) == 2
    assert queue._events[0][0] <= queue._events[1][0]

def test_aggregate():
    event_list = [(0.2 * i, i) for i in range(5)]

    queue = eventqueue.EventQueue(event_list, min_resolution=0.3)

    assert queue.pop() == (0.3, [0, 1])
    assert queue.pop() == (0.6, [2])
    assert queue.pop() == (0.8999999999999999, [3, 4])
