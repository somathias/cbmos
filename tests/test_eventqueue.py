import cbmos.cbmodel._eventqueue as eventqueue
import cbmos.cell as cl
import cbmos.events as ev

def test_constructor():
    separation = 1.
    cells = [cl.Cell(i, [0, 0, i], proliferating=True, division_time=i) for i in range(5)]

    event_list = [ev.CellDivisionEvent(cells[i]) for i in reversed(range(5))]

    queue = eventqueue.EventQueue(event_list)

    assert len(queue._events) == 5

    for i in range(5):
        t, event = queue.pop()
        assert t == i
        assert len(event) == 1
        assert event[0].target_cell_ID == i

def test_push():
    cell = cl.Cell(0, [0, 0], proliferating=True, division_time=1)

    queue = eventqueue.EventQueue([])
    event = ev.CellDivisionEvent(cell)
    queue.push(event)

    assert len(queue._events) == 1

    assert (1, event) == queue._events[0]

    cell2 = cl.Cell(1, [0.25, 0.25], proliferating=True, division_time=2)
    event2 = ev.CellDivisionEvent(cell2)
    queue.push(event2)

    assert len(queue._events) == 2
    assert queue._events[0][0] <= queue._events[1][0]

def test_aggregate():
    event_list = [
        ev.CellDivisionEvent(cl.Cell(i, [0, 0, i], division_time=0.2*i, proliferating=True))
        for i in range(5)]

    queue = eventqueue.EventQueue(event_list, min_resolution=0.3)

    for true_t, true_ids in [(0.3, [0, 1]), (0.6, [2]), (0.8999999999999999, [3, 4])]:
        t, events = queue.pop()
        assert t == true_t and [e.target_cell_ID for e in events] == true_ids
