"""
WebSocket Handlers
==================
WebSocket endpoints for real-time features.
"""

import logging
from typing import Optional
import uuid
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from jose import jwt, JWTError

from app.websocket.manager import manager
from app.config import settings
from app.services.redis_service import redis_service

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_user_from_token(token: str) -> Optional[int]:
    """Extract user ID from JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id = payload.get("sub")
        return int(user_id) if user_id else None
    except JWTError:
        return None


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """Main WebSocket endpoint for real-time updates."""
    connection_id = str(uuid.uuid4())
    user_id = None
    
    if token:
        user_id = await get_user_from_token(token)
    
    await manager.connect(websocket, connection_id, user_id)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "connection_id": connection_id,
            "authenticated": user_id is not None,
        })
        
        while True:
            data = await websocket.receive_json()
            await handle_message(connection_id, user_id, data)
    
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(connection_id)


async def handle_message(
    connection_id: str,
    user_id: Optional[int],
    data: dict,
) -> None:
    """Handle incoming WebSocket messages."""
    message_type = data.get("type")
    
    if message_type == "subscribe":
        channel = data.get("channel")
        if channel:
            await manager.subscribe(connection_id, channel)
            await manager.send_personal_message(
                {"type": "subscribed", "channel": channel},
                connection_id,
            )
    
    elif message_type == "unsubscribe":
        channel = data.get("channel")
        if channel:
            await manager.unsubscribe(connection_id, channel)
            await manager.send_personal_message(
                {"type": "unsubscribed", "channel": channel},
                connection_id,
            )
    
    elif message_type == "ping":
        await manager.send_personal_message(
            {"type": "pong", "timestamp": data.get("timestamp")},
            connection_id,
        )
    
    elif message_type == "get_latest_price":
        # Get latest price from cache
        tick = redis_service.get("xauusd:tick:latest")
        await manager.send_personal_message(
            {"type": "price", "data": tick},
            connection_id,
        )
    
    elif message_type == "get_latest_signal":
        # Get latest signal from cache
        signal = redis_service.get("signal:latest")
        await manager.send_personal_message(
            {"type": "signal", "data": signal},
            connection_id,
        )
    
    elif message_type == "get_latest_prediction":
        # Get latest prediction from cache
        prediction = redis_service.get("prediction:latest")
        await manager.send_personal_message(
            {"type": "prediction", "data": prediction},
            connection_id,
        )


@router.websocket("/ws/prices")
async def price_stream(websocket: WebSocket):
    """Dedicated price streaming endpoint."""
    connection_id = str(uuid.uuid4())
    await manager.connect(websocket, connection_id)
    await manager.subscribe(connection_id, "prices")
    
    try:
        # Start price streaming loop
        while True:
            # Check for client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1,
                )
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                pass
            
            # Get and send latest price
            tick = redis_service.get("xauusd:tick:latest")
            if tick:
                await websocket.send_json({
                    "type": "tick",
                    "data": tick,
                })
            
            await asyncio.sleep(0.5)  # Update every 500ms
    
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"Price stream error: {e}")
        await manager.disconnect(connection_id)


@router.websocket("/ws/signals")
async def signal_stream(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """Dedicated signal streaming endpoint."""
    connection_id = str(uuid.uuid4())
    user_id = await get_user_from_token(token) if token else None
    
    await manager.connect(websocket, connection_id, user_id)
    await manager.subscribe(connection_id, "signals")
    
    try:
        # Send current signal on connect
        signal = redis_service.get("signal:latest")
        if signal:
            await websocket.send_json({
                "type": "signal",
                "data": signal,
            })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"Signal stream error: {e}")
        await manager.disconnect(connection_id)


@router.websocket("/ws/trades")
async def trade_updates(
    websocket: WebSocket,
    token: str = Query(...),
):
    """Real-time trade updates for authenticated users."""
    user_id = await get_user_from_token(token)
    
    if not user_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    connection_id = str(uuid.uuid4())
    await manager.connect(websocket, connection_id, user_id)
    await manager.subscribe(connection_id, f"trades:{user_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"Trade updates error: {e}")
        await manager.disconnect(connection_id)
