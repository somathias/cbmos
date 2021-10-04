import heapq as _hq

class EventQueue:
    """
    Parameters
    ----------
        events: Event
            List of events.
        min_resolution: float
            time resolution at which events can happen
    """

    def __init__(self, events, min_resolution=0.):
        assert min_resolution >= 0.

        self._min_resolution = min_resolution
        events = [
                (self._round_time(event.tau), event)
                for event in events
                ]
        _hq.heapify(events)
        self._events = events

    def push(self, event):
        """
        Add one event into the queue.

        Parameters
        ----------
            event: (float, Event)
                Event to be added to the queue
        """
        _hq.heappush(self._events, (self._round_time(event.tau), event))

    def pop(self):
        """
        Returns the next events to occur. If several events occur at the same
        time, they are returned together.
        Returns
        -------
            (float, [Event])
                Time at which the events occur and a list of these events
        """
        next_event_time = self._events[0][0]
        next_events = []
        while len(self._events) > 0 and self._events[0][0] == next_event_time:
            next_events.append(_hq.heappop(self._events)[1])

        return (next_event_time, next_events)

    def _round_time(self, t):
        if self._min_resolution > 0.0:
            return (t//self._min_resolution + 1) * (self._min_resolution)
        else:
            return t
