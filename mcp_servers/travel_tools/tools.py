from app.graph.state import DestinationInfo

DESTINATIONS: dict[str, DestinationInfo] = {
    "hanoi": {
        "destination": "Hanoi",
        "found": True,
        "summary": "A historic Vietnamese city known for its Old Quarter and lakes.",
        "activities": ["Explore the Old Quarter", "Visit Hoan Kiem Lake"],
    },
    "kyoto": {
        "destination": "Kyoto",
        "found": True,
        "summary": "A Japanese city known for historic temples and traditional districts.",
        "activities": ["Visit a historic temple", "Walk through Gion"],
    },
}


def search_destination_data(destination: str) -> DestinationInfo:
    normalized = destination.strip().casefold()
    result = DESTINATIONS.get(normalized)
    if result is None:
        return {
            "destination": destination.strip(),
            "found": False,
            "summary": "",
            "activities": [],
        }
    return {
        "destination": result["destination"],
        "found": result["found"],
        "summary": result["summary"],
        "activities": list(result["activities"]),
    }
