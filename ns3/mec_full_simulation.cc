#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"
#include "ns3/mobility-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-routing-table-entry.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/socket-factory.h"
#include <fstream>
#include <iomanip>


#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include <cstdio>
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <sstream>

#include "ns3/udp-client.h"
#include "ns3/socket.h"
#include "ns3/core-module.h"
#include "ns3/application.h"
#include "ns3/inet-socket-address.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/random-variable-stream.h"

using namespace ns3;

// Forward declarations before RLSocketInterface
std::map<uint16_t, Ptr<Socket>> g_portToSocket;           // destPort → Socket
std::map<uint16_t, Ptr<OnOffApplication>> g_portToOnOffApp; // destPort → OnOffApp
NetDeviceContainer indiaRouterToGNodeB, indiaGNodeBToQoS, indiaQoSToMEC;
NetDeviceContainer indiaMECToUPF, indiaUPFToBackbone;
NetDeviceContainer ukRouterToGNodeB, ukGNodeBToQoS, ukQoSToMEC;
NetDeviceContainer ukMECToUPF, ukUPFToBackbone;
NetDeviceContainer backboneLink, backboneToCloud, backboneToInternet;

// CWND TRACKING
// ============================================================================
// Store CWND by flow 5-tuple (more reliable than socket pointer)
struct FlowKey {
    Ipv4Address srcAddr;
    Ipv4Address dstAddr;
    uint16_t srcPort;
    uint16_t dstPort;
    uint8_t protocol;
    
    bool operator<(const FlowKey& other) const {
        if (srcAddr != other.srcAddr) return srcAddr < other.srcAddr;
        if (dstAddr != other.dstAddr) return dstAddr < other.dstAddr;
        if (srcPort != other.srcPort) return srcPort < other.srcPort;
        if (dstPort != other.dstPort) return dstPort < other.dstPort;
        return protocol < other.protocol;
    }

     bool operator==(const FlowKey& other) const {
        return srcAddr == other.srcAddr &&
               dstAddr == other.dstAddr &&
               srcPort == other.srcPort &&
               dstPort == other.dstPort &&
               protocol == other.protocol;
    }

    bool operator!=(const FlowKey& other) const {
        return !(*this == other);
    }
};
std::map<FlowKey, uint32_t> g_flowCwndByTuple;

// ✅ NEW: Store FlowID → Port mapping (updated during metrics collection)
std::map<FlowId, uint16_t> g_flowIdToPort;

class RLSocketInterface
{
public:
    RLSocketInterface(std::string aiHost = "127.0.0.1", uint16_t aiPort = 5000)
        : m_aiHost(aiHost),
          m_aiPort(aiPort),
          m_socket(-1),
          m_socketReady(false),
          m_messagesSent(0),
          m_messagesReceived(0),
          m_errors(0)
    {
    }
    
    ~RLSocketInterface()
    {
        Close();
    }
    
    bool Initialize()
    {
        // Create OS-level UDP socket
        m_socket = socket(AF_INET, SOCK_DGRAM, 0);
        if (m_socket < 0) {
            std::cerr << "❌ Failed to create socket: " << strerror(errno) << std::endl;
            return false;
        }
        
        // Set non-blocking + timeout
        int flags = fcntl(m_socket, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(m_socket, F_SETFL, flags | O_NONBLOCK);
        }
        
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50000;  // 50ms timeout
        setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        
        // Configure AI controller address
        memset(&m_aiAddr, 0, sizeof(m_aiAddr));
        m_aiAddr.sin_family = AF_INET;
        m_aiAddr.sin_port = htons(m_aiPort);
        
        if (inet_pton(AF_INET, m_aiHost.c_str(), &m_aiAddr.sin_addr) <= 0) {
            std::cerr << "❌ Invalid AI controller address: " << m_aiHost << std::endl;
            close(m_socket);
            m_socket = -1;
            return false;
        }
        
        m_socketReady = true;
        std::cout << "✓ RL Socket initialized: " << m_aiHost << ":" << m_aiPort << std::endl;
        
        // Send handshake
        std::ostringstream handshake;
        handshake << "{"
                  << "\"type\":\"handshake\","
                  << "\"version\":\"ns-3.45\","
                  << "\"message\":\"MEC Simulation Ready\","
                  << "\"agents\":[\"ppo_flow\",\"a3c_edge\",\"gwo_resource\"],"
                  << "\"timestamp\":" << Simulator::Now().GetSeconds()
                  << "}";
        
        SendMessage(handshake.str());
        
        return true;
    }
    
    void SendMetrics(const std::string& metricsJson)
    {
        if (!m_socketReady) return;
        
        SendMessage(metricsJson);
        
        // ✅ NEW: Wait up to 500ms for Python response
        char buffer[8192];
        
        // Set receive timeout (if not already set globally)
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 500000;  // 500ms - plenty of time for Python to respond
        setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        
        ssize_t received = recvfrom(m_socket, buffer, 8191, 0,  // ← Remove MSG_DONTWAIT!
                                    nullptr, nullptr);
        
        if (received > 0) {
            buffer[received] = '\0';
            m_messagesReceived++;
            m_lastAction = std::string(buffer);
            
            std::cout << "✅ Received action from AI (" << received << " bytes)" << std::endl;
            
            // Parse and apply action
            ApplyAction(m_lastAction);
        } 
        else if (received == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Timeout - Python didn't respond (this is OK early in simulation)
                if (Simulator::Now().GetSeconds() < 5.0) {
                    std::cout << "⏳ No action received yet (t=" 
                            << Simulator::Now().GetSeconds() 
                            << "s - flows may not have started)" << std::endl;
                } else {
                    std::cout << "⚠️  Action timeout (Python may be slow or crashed)" << std::endl;
                }
            } else {
                // Actual error
                std::cerr << "❌ Socket error: " << strerror(errno) << std::endl;
                m_errors++;
            }
        }
    }
    
    void Close()
    {
        if (m_socket >= 0) {
            std::ostringstream goodbye;
            goodbye << "{"
                    << "\"type\":\"goodbye\","
                    << "\"stats\":{"
                    << "\"messages_sent\":" << m_messagesSent << ","
                    << "\"messages_received\":" << m_messagesReceived << ","
                    << "\"errors\":" << m_errors
                    << "}}";
            
            SendMessage(goodbye.str());
            close(m_socket);
            m_socket = -1;
            m_socketReady = false;
            
            std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
            std::cout << "║      RL SOCKET STATISTICS                ║" << std::endl;
            std::cout << "╠══════════════════════════════════════════╣" << std::endl;
            std::cout << "║  Messages Sent:     " << std::setw(10) << m_messagesSent << "         ║" << std::endl;
            std::cout << "║  Actions Received:  " << std::setw(10) << m_messagesReceived << "         ║" << std::endl;
            std::cout << "║  Errors:            " << std::setw(10) << m_errors << "         ║" << std::endl;
            std::cout << "╚══════════════════════════════════════════╝\n" << std::endl;
        }
    }
    
    bool IsReady() const { return m_socketReady; }
    std::string GetLastAction() const { return m_lastAction; }

private:
    void SendMessage(const std::string& msg)
    {
        ssize_t sent = sendto(m_socket, msg.c_str(), msg.length(), 0,
                              (struct sockaddr*)&m_aiAddr, sizeof(m_aiAddr));
        
        if (sent > 0) {
            m_messagesSent++;
        } else {
            m_errors++;
            std::cerr << "⚠ Failed to send message: " << strerror(errno) << std::endl;
        }
    }

private:
    void SendActionAck(const std::string& actionId, bool success, const std::string& details)
    {
        std::ostringstream ack;
        ack << "{"
            << "\"type\":\"action_ack\","
            << "\"action_id\":\"" << actionId << "\","
            << "\"success\":" << (success ? "true" : "false") << ","
            << "\"details\":\"" << details << "\""
            << "}";
            
        SendMessage(ack.str());
    }

    // In RLSocketInterface::ApplyAction():
    void ApplyAction(const std::string& actionJson)
    {
        rapidjson::Document doc;
        doc.Parse(actionJson.c_str());
        
        if (doc.HasParseError()) {
            std::cerr << "❌ JSON parse error\n";
            return;
        }
        // Extract action_id if present
        std::string actionId = "";
        if (doc.HasMember("action_id")) {
            actionId = doc["action_id"].GetString();
        }

        bool success = false;
        std::string details;

        if (doc.HasMember("flow_actions")) {
            const rapidjson::Value& flowActions = doc["flow_actions"];
            // Track success of all flow actions
            int successfulActions = 0;
            int totalActions = flowActions.Size();
            
            
            for (rapidjson::SizeType i = 0; i < flowActions.Size(); ++i) {
                const rapidjson::Value& action = flowActions[i];
                
                if (!action.HasMember("flow_id") || !action.HasMember("action")) {
                    continue;
                }
                
                uint32_t flowId = action["flow_id"].GetInt();
                std::string actionType = action["action"].GetString();
                
                std::cout << "✓ Applying action for Flow " << flowId << ": " << actionType << "\n";
                
                auto flowPortIt = g_flowIdToPort.find(flowId);
                if (flowPortIt == g_flowIdToPort.end()) {
                    std::cerr << "  ⚠️  Flow " << flowId << " not found in port mapping\n";
                    continue;
                }
                
                uint16_t destPort = flowPortIt->second;
                auto portIt = g_portToSocket.find(destPort);
                
                if (portIt != g_portToSocket.end()) {
                    Ptr<Socket> socket = portIt->second;
                    Ptr<TcpSocket> tcpSocket = DynamicCast<TcpSocket>(socket);
                    bool actionSuccess = false;
                    
                try{
                        // if (actionType == "INCREASE_PRIORITY") {
                        //     if (tcpSocket) {
                        //         // Use Socket's attribute system instead of direct CWND access
                        //         DoubleValue cwnd;
                        //         tcpSocket->GetAttribute("SlowStartThreshold", cwnd);
                        //         uint32_t newCwnd = std::min((uint32_t)(cwnd.Get() * 2), (uint32_t)20);
                                
                        //         // Set new values through attributes
                        //         tcpSocket->SetAttribute("SlowStartThreshold", UintegerValue(newCwnd));
                        //         tcpSocket->SetAttribute("InitialCwnd", UintegerValue(newCwnd));
                                
                        //         // Set IP TOS for priority
                        //         socket->SetIpTos(0xb8);
                        //         actionSuccess = true;
                        //         details += "Increased CWND to " + std::to_string(newCwnd) + 
                        //                 " for flow " + std::to_string(flowId) + "; ";
                        //     }
                        // }
                        // else if (actionType == "DECREASE_PRIORITY") {
                        //     if (tcpSocket) {
                        //         DoubleValue cwnd;
                        //         tcpSocket->GetAttribute("SlowStartThreshold", cwnd);
                        //         uint32_t newCwnd = std::max((uint32_t)(cwnd.Get() / 2), (uint32_t)5);
                                
                        //         tcpSocket->SetAttribute("SlowStartThreshold", UintegerValue(newCwnd));
                        //         tcpSocket->SetAttribute("InitialCwnd", UintegerValue(newCwnd));
                                
                        //         socket->SetIpTos(0x00);
                        //         actionSuccess = true;
                        //         details += "Decreased CWND to " + std::to_string(newCwnd) + 
                        //                 " for flow " + std::to_string(flowId) + "; ";
                        //     }
                        // }
                        if (actionType == "INCREASE_RATE") {
                            auto appIt = g_portToOnOffApp.find(destPort);
                            if (appIt != g_portToOnOffApp.end()) {
                                Ptr<OnOffApplication> app = appIt->second;
                                DataRateValue currentRate;
                                app->GetAttribute("DataRate", currentRate);
                                
                                uint64_t currentBps = currentRate.Get().GetBitRate();
                                uint64_t newRateBps = currentBps * 1.2;
                                
                                // ✅ Enforce rate limits (1 Kbps to 500 Mbps)
                                const uint64_t MIN_RATE = 10000;      // 10 Kbps
                                const uint64_t MAX_RATE = 500000000; // 500 Mbps
                                
                                newRateBps = std::max(newRateBps, MIN_RATE);
                                newRateBps = std::min(newRateBps, MAX_RATE);
                                
                                if (newRateBps != currentBps) {
                                    app->SetAttribute("DataRate", DataRateValue(DataRate(newRateBps)));
                                    actionSuccess = true;
                                    std::cout << "✅ Flow " << flowId << ": Increased rate to " 
                                        << newRateBps/1e6 << " Mbps" << std::endl;
                                    details += "Increased rate to " + std::to_string(newRateBps/1e6) + 
                                            " Mbps for flow " + std::to_string(flowId) + "; ";
                                } else {
                                    details += "Rate already at limit for flow " + std::to_string(flowId) + "; ";
                                }
                            }
                        }
                        else if (actionType == "DECREASE_RATE") {
                            auto appIt = g_portToOnOffApp.find(destPort);
                            if (appIt != g_portToOnOffApp.end()) {
                                Ptr<OnOffApplication> app = appIt->second;
                                DataRateValue currentRate;
                                app->GetAttribute("DataRate", currentRate);
                                
                                uint64_t currentBps = currentRate.Get().GetBitRate();
                                uint64_t newRateBps = currentBps * 0.8;
                                
                                const uint64_t MIN_RATE = 10000;
                                const uint64_t MAX_RATE = 500000000;
                                
                                newRateBps = std::max(newRateBps, MIN_RATE);
                                newRateBps = std::min(newRateBps, MAX_RATE);
                                
                                if (newRateBps != currentBps) {
                                    app->SetAttribute("DataRate", DataRateValue(DataRate(newRateBps)));
                                    actionSuccess = true;
                                    details += "Decreased rate to " + std::to_string(newRateBps/1e6) + 
                                            " Mbps for flow " + std::to_string(flowId) + "; ";
                                } else {
                                    details += "Rate already at minimum for flow " + std::to_string(flowId) + "; ";
                                }
                            }
                        }
                        else if (actionType == "INCREASE_BURST_SIZE") {
                            auto appIt = g_portToOnOffApp.find(destPort);
                            if (appIt != g_portToOnOffApp.end()) {
                                Ptr<OnOffApplication> app = appIt->second;
                                UintegerValue currentSize;
                                app->GetAttribute("PacketSize", currentSize);
                                
                                // ✅ CRITICAL FIX: Enforce maximum packet size
                                uint32_t newSize = currentSize.Get() * 1.5;
                                
                                const uint32_t MIN_PACKET_SIZE = 64;
                                const uint32_t MAX_PACKET_SIZE = 1500;
                                
                                newSize = std::max(newSize, MIN_PACKET_SIZE);
                                newSize = std::min(newSize, MAX_PACKET_SIZE);
                                
                                // Only apply if change is meaningful
                                if (newSize != currentSize.Get()) {
                                    app->SetAttribute("PacketSize", UintegerValue(newSize));
                                    actionSuccess = true;
                                    details += "Increased burst size to " + std::to_string(newSize) + 
                                            " bytes for flow " + std::to_string(flowId) + "; ";
                                    std::cout << "✅ Flow " << flowId << ": Increased burst size to " 
                                            << newSize << " bytes" << std::endl;
                                } else {
                                    details += "Burst size already at maximum (" + std::to_string(MAX_PACKET_SIZE) + 
                                            " bytes) for flow " + std::to_string(flowId) + "; ";
                                    std::cout << "⚠️ Flow " << flowId << ": Already at maximum burst size" << std::endl;
                                }
                            }
                        }
                        else if (actionType == "DECREASE_BURST_SIZE") {
                            auto appIt = g_portToOnOffApp.find(destPort);
                            if (appIt != g_portToOnOffApp.end()) {
                                Ptr<OnOffApplication> app = appIt->second;
                                UintegerValue currentSize;
                                app->GetAttribute("PacketSize", currentSize);
                                
                                // ✅ CRITICAL FIX: Enforce minimum packet size
                                uint32_t newSize = currentSize.Get() * 0.75;
                                
                                // Minimum viable packet size: 64 bytes (Ethernet minimum)
                                // Maximum: 1500 bytes (typical MTU)
                                const uint32_t MIN_PACKET_SIZE = 64;
                                const uint32_t MAX_PACKET_SIZE = 1500;
                                
                                newSize = std::max(newSize, MIN_PACKET_SIZE);
                                newSize = std::min(newSize, MAX_PACKET_SIZE);
                                
                                // Only apply if change is meaningful
                                if (newSize != currentSize.Get()) {
                                    app->SetAttribute("PacketSize", UintegerValue(newSize));
                                    actionSuccess = true;
                                    details += "Decreased burst size to " + std::to_string(newSize) + 
                                            " bytes for flow " + std::to_string(flowId) + "; ";
                                    std::cout << "✅ Flow " << flowId << ": Decreased burst size to " 
                                            << newSize << " bytes" << std::endl;
                                } else {
                                    details += "Burst size already at minimum (" + std::to_string(MIN_PACKET_SIZE) + 
                                            " bytes) for flow " + std::to_string(flowId) + "; ";
                                    std::cout << "⚠️ Flow " << flowId << ": Already at minimum burst size" << std::endl;
                                }
                            }
                        }
                        // else if (actionType == "SET_MIN_RTO") {
                        //     socket->SetAttribute("MinRto", TimeValue(MilliSeconds(200)));
                        //     std::cout << "  → Set minimum RTO to 200ms (port " << destPort << ")\n";
                        // }
                        // else if (actionType == "SET_MAX_RTO") {
                        //     socket->SetAttribute("MaxRto", TimeValue(Seconds(2)));
                        //     std::cout << "  → Set maximum RTO to 2s (port " << destPort << ")\n";
                        // }
                        else if (actionType == "NO_CHANGE") {
                            actionSuccess = true;
                            details += "No change for flow " + std::to_string(flowId) + "; ";
                        }
                        else {
                            std::cerr << "  ⚠️  Unknown action type: " << actionType << "\n";
                            details += "Unknown action type: " + actionType + "; ";
                        }
                        if (actionSuccess) {
                        successfulActions++;
                        std::cout << "💫 Action executed: " << actionType 
                            << " for Flow " << flowId 
                            << " (Port " << destPort << ")" << std::endl;
                    } else {
                        std::cout << "⚠️  Action failed: " << actionType 
                                << " for Flow " << flowId 
                                << " - " << details << std::endl;
                    }
                        
                    } 
                    
                    catch (const std::exception& e) {
                        std::cerr << "  ⚠️  Error applying action: " << e.what() << "\n";
                        details += "Error: " + std::string(e.what()) + "; ";
                    }

                    
                }
            }
            // Consider overall success if majority of actions succeeded
            success = (successfulActions > totalActions / 2);
            
            if (success) {
                details = "Successfully applied " + std::to_string(successfulActions) + 
                        " out of " + std::to_string(totalActions) + " flow actions";
            } else {
                details = "Failed to apply majority of flow actions (" + 
                        std::to_string(successfulActions) + "/" + 
                        std::to_string(totalActions) + ")";
            }
        }

        // // Send acknowledgment
        // SendActionAck(actionId, success, details);

        // ✅ NEW: Handle A3C edge actions - UPDATED FOR RESOURCE CONTROL
        if (doc.HasMember("edge_actions")) {
            const rapidjson::Value& edgeActions = doc["edge_actions"];
            
            for (rapidjson::SizeType i = 0; i < edgeActions.Size(); ++i) {
                const rapidjson::Value& action = edgeActions[i];
                
                if (!action.HasMember("location") || !action.HasMember("action")) {
                    continue;
                }
                
                std::string location = action["location"].GetString();
                std::string actionType = action["action"].GetString();
                
                std::cout << "✓ A3C Edge action at " << location << ": " << actionType << "\n";
                
                // ========== A3C EDGE RESOURCE CONTROL ACTIONS ==========
                
                if (actionType == "ADJUST_QUEUE_SIZE") {
                    // Dynamic queue sizing - affects ALL traffic types
                    NetDeviceContainer* targetLink = nullptr;
                    
                    if (location == "india") {
                        targetLink = &indiaRouterToGNodeB;
                    } else if (location == "uk") {
                        targetLink = &ukRouterToGNodeB;
                    }
                    
                    if (targetLink) {
                        Ptr<NetDevice> device = targetLink->Get(0);
                        Ptr<QueueDisc> qdisc = device->GetNode()->GetObject<TrafficControlLayer>()
                                                    ->GetRootQueueDiscOnDevice(device);
                        
                        if (qdisc) {
                            uint32_t currentMax = qdisc->GetMaxSize().GetValue();
                            uint32_t newMax = currentMax;
                            
                            // Get target size from parameters if provided
                            if (action.HasMember("queue_size")) {
                                newMax = action["queue_size"].GetInt();
                            } else {
                                // Default adjustment logic
                                newMax = (actionType == "ADJUST_QUEUE_SIZE") ? 
                                        std::min(currentMax + 20, (uint32_t)200) : 
                                        std::max(currentMax - 15, (uint32_t)50);
                            }
                            
                            // Clamp to valid range
                            newMax = std::min(std::max(newMax, (uint32_t)50), (uint32_t)200);
                            qdisc->SetMaxSize(QueueSize(QueueSizeUnit::PACKETS, newMax));
                            
                            std::cout << "   📊 Adjusted queue size: " << currentMax 
                                    << " → " << newMax << " packets\n";
                            
                            success = true;
                            details += "Queue size: " + std::to_string(newMax) + " packets; ";
                        }
                    }
                }
                
                else if (actionType == "SET_URLLC_RATE_LIMIT") {
                    // Rate limiting for URLLC traffic (prevents UDP starvation)
                    double rateLimitMbps = 100.0; // Default limit
                    
                    if (action.HasMember("rate_limit_mbps")) {
                        rateLimitMbps = action["rate_limit_mbps"].GetDouble();
                    }
                    
                    // Apply rate limiting logic (simulated)
                    // In practice, this would configure traffic shapers
                    std::cout << "   🚦 URLLC Rate Limit: " << rateLimitMbps << " Mbps\n";
                    
                    // Find URLLC flows and apply limits
                    for (auto& [port, onoffApp] : g_portToOnOffApp) {
                        // Identify URLLC flows by port range (10-13)
                        if (port >= 10 && port <= 13) {
                            DataRateValue currentRate;
                            onoffApp->GetAttribute("DataRate", currentRate);
                            uint64_t currentBps = currentRate.Get().GetBitRate();
                            uint64_t limitBps = rateLimitMbps * 1000000;
                            
                            if (currentBps > limitBps) {
                                onoffApp->SetAttribute("DataRate", 
                                    DataRateValue(DataRate(limitBps)));
                                std::cout << "   ⚡ Limited URLLC flow (port " << port 
                                        << ") to " << rateLimitMbps << " Mbps\n";
                            }
                        }
                    }
                    
                    success = true;
                    details += "URLLC rate limit: " + std::to_string(rateLimitMbps) + " Mbps; ";
                }
                
                else if (actionType == "ADJUST_PRIORITY_WEIGHTS") {
                    // Modify priority scheduling weights for network slicing
                    std::cout << "   ⚖️ Adjusting Radio Slice Weights\n";
                    
                    NetDeviceContainer* targetLink = nullptr;
                    if (location == "india") targetLink = &indiaGNodeBToQoS;
                    else if (location == "uk") targetLink = &ukGNodeBToQoS;
                    
                    if (targetLink) {
                        Ptr<NetDevice> device = targetLink->Get(0);
                        Ptr<TrafficControlLayer> tc = device->GetNode()->GetObject<TrafficControlLayer>();
                        Ptr<QueueDisc> qdisc = tc->GetRootQueueDiscOnDevice(device);
                        
                        if (qdisc && qdisc->GetInstanceTypeId().GetName() == "ns3::PriomapQueueDisc") {
                            // Get new slice rates from parameters
                            DataRate urllcRate = DataRate("50Mbps");  // default
                            DataRate embbRate = DataRate("100Mbps");  // default  
                            DataRate mmtcRate = DataRate("20Mbps");   // default
                            
                            if (action.HasMember("urllc_rate_mbps")) {
                                double rate = action["urllc_rate_mbps"].GetDouble();
                                urllcRate = DataRate(std::to_string((int)rate) + "Mbps");
                            }
                            if (action.HasMember("embb_rate_mbps")) {
                                double rate = action["embb_rate_mbps"].GetDouble();
                                embbRate = DataRate(std::to_string((int)rate) + "Mbps");
                            }
                            if (action.HasMember("mmtc_rate_mbps")) {
                                double rate = action["mmtc_rate_mbps"].GetDouble();
                                mmtcRate = DataRate(std::to_string((int)rate) + "Mbps");
                            }
                            
                            // Update child queue disc rates
                            Ptr<QueueDisc> urllcQueue = qdisc->GetQueueDiscClass(0)->GetQueueDisc();
                            Ptr<QueueDisc> embbQueue = qdisc->GetQueueDiscClass(1)->GetQueueDisc();
                            Ptr<QueueDisc> mmtcQueue = qdisc->GetQueueDiscClass(2)->GetQueueDisc();
                            
                            if (urllcQueue) {
                                urllcQueue->SetAttribute("Rate", DataRateValue(urllcRate));
                                urllcQueue->SetAttribute("PeakRate", DataRateValue(DataRate(urllcRate.GetBitRate() * 1.2)));
                            }
                            if (embbQueue) {
                                embbQueue->SetAttribute("Rate", DataRateValue(embbRate));
                                embbQueue->SetAttribute("PeakRate", DataRateValue(DataRate(embbRate.GetBitRate() * 1.2)));
                            }
                            if (mmtcQueue) {
                                mmtcQueue->SetAttribute("Rate", DataRateValue(mmtcRate));
                                mmtcQueue->SetAttribute("PeakRate", DataRateValue(DataRate(mmtcRate.GetBitRate() * 1.5)));
                            }
                            
                            std::cout << "   🔄 Updated Radio Slice Rates:\n";
                            std::cout << "      - URLLC: " << urllcRate << "\n";
                            std::cout << "      - eMBB:  " << embbRate << "\n";
                            std::cout << "      - mMTC:  " << mmtcRate << "\n";
                            
                            details += "URLLC:" + std::to_string(urllcRate.GetBitRate()/1e6) + "Mbps ";
                            details += "eMBB:" + std::to_string(embbRate.GetBitRate()/1e6) + "Mbps ";
                            details += "mMTC:" + std::to_string(mmtcRate.GetBitRate()/1e6) + "Mbps; ";
                            
                            success = true;
                        } else {
                            std::cout << "   ⚠️  No PriomapQueueDisc found for radio scheduling\n";
                            details += "No radio scheduling queue found; ";
                        }
                    }
                }
                
                else if (actionType == "ENABLE_ADMISSION_CONTROL") {
                    // Simulate admission control by limiting new flow creation
                    bool enable = true;
                    if (action.HasMember("enable")) {
                        enable = action["enable"].GetBool();
                    }
                    
                    if (enable) {
                        std::cout << "   🛑 Admission control ENABLED\n";
                        // In practice, this would block new flow setups during congestion
                        details += "Admission control enabled; ";
                    } else {
                        std::cout << "   ✅ Admission control DISABLED\n";
                        details += "Admission control disabled; ";
                    }
                    
                    success = true;
                }
                
                else if (actionType == "TUNE_AQM_PARAMETERS") {
                    // Adjust Active Queue Management parameters
                    NetDeviceContainer* targetLink = nullptr;
                    if (location == "india") targetLink = &indiaRouterToGNodeB;
                    else if (location == "uk") targetLink = &ukRouterToGNodeB;
                    
                    if (targetLink) {
                        Ptr<NetDevice> device = targetLink->Get(0);
                        Ptr<QueueDisc> qdisc = device->GetNode()->GetObject<TrafficControlLayer>()
                                                    ->GetRootQueueDiscOnDevice(device);
                        
                        if (qdisc) {
                            std::string qdiscType = qdisc->GetInstanceTypeId().GetName();
                            
                            if (qdiscType.find("CoDel") != std::string::npos) {
                                // Adjust CoDel parameters
                                TimeValue interval, target;
                                qdisc->GetAttribute("Interval", interval);
                                qdisc->GetAttribute("Target", target);
                                
                                // Example: Reduce target delay for URLLC sensitivity
                                if (action.HasMember("target_delay_ms")) {
                                    double newTargetMs = action["target_delay_ms"].GetDouble();
                                    Time newTarget = MilliSeconds(newTargetMs);
                                    qdisc->SetAttribute("Target", TimeValue(newTarget));
                                    
                                    std::cout << "   🎯 CoDel Target: " << target.Get().GetMilliSeconds() 
                                            << "ms → " << newTargetMs << "ms\n";
                                    details += "CoDel target: " + std::to_string(newTargetMs) + "ms; ";
                                }
                                
                                if (action.HasMember("interval_ms")) {
                                    double newIntervalMs = action["interval_ms"].GetDouble();
                                    Time newInterval = MilliSeconds(newIntervalMs);
                                    qdisc->SetAttribute("Interval", TimeValue(newInterval));
                                    
                                    std::cout << "   ⏱️ CoDel Interval: " << interval.Get().GetMilliSeconds() 
                                            << "ms → " << newIntervalMs << "ms\n";
                                    details += "CoDel interval: " + std::to_string(newIntervalMs) + "ms; ";
                                }
                            }
                        }
                    }
                    
                    success = true;
                }
                
                else if (actionType == "NO_CHANGE") {
                    success = true;
                    details += "No edge changes; ";
                }
                
                else {
                    std::cerr << "  ⚠️  Unknown edge action: " << actionType << "\n";
                    details += "Unknown action: " + actionType + "; ";
                }
            }
        }

        // ========== GWO GLOBAL RESOURCE ALLOCATION ==========
        // ✅ NEW: Handle GWO backbone bandwidth optimization
        if (doc.HasMember("gwo_actions")) {
            const rapidjson::Value& gwoActions = doc["gwo_actions"];
            
            if (gwoActions.HasMember("allocation")) {
                const rapidjson::Value& alloc = gwoActions["allocation"];
                
                std::cout << "\n🐺 GWO RESOURCE ALLOCATION RECEIVED\n";
                std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                
                // ===== PRIMARY: Backbone Bandwidth Adjustment =====
                if (alloc.HasMember("backbone_bandwidth")) {
                    double bw_ratio = alloc["backbone_bandwidth"].GetDouble();
                    
                    // Convert ratio (0.3-1.0) to actual bandwidth (60-200 Mbps)
                    uint64_t base_bw = 200000000;  // 200 Mbps in bps
                    uint64_t min_bw = 60000000;    // 60 Mbps minimum
                    uint64_t new_bw = min_bw + (uint64_t)((base_bw - min_bw) * bw_ratio);
                    
                    // Apply to backbone link (both directions)
                    try {
                        Ptr<PointToPointNetDevice> dev0 = 
                            DynamicCast<PointToPointNetDevice>(backboneLink.Get(0));
                        Ptr<PointToPointNetDevice> dev1 = 
                            DynamicCast<PointToPointNetDevice>(backboneLink.Get(1));
                        
                        if (dev0 && dev1) {
                            dev0->SetDataRate(DataRate(new_bw));
                            dev1->SetDataRate(DataRate(new_bw));
                            
                            std::cout << "  ✓ Backbone Bandwidth Changed:\n";
                            std::cout << "    Ratio: " << bw_ratio << " (" 
                                     << (bw_ratio * 100) << "%)\n";
                            std::cout << "    New BW: " << (new_bw / 1e6) << " Mbps\n";
                            
                            success = true;
                            details += "Backbone BW: " + std::to_string(new_bw/1e6) + " Mbps; ";
                        } else {
                            std::cerr << "  ⚠️  Failed to get backbone devices\n";
                            details += "Backbone BW change failed; ";
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "  ⚠️  Error changing backbone BW: " << e.what() << "\n";
                        details += "Error: " + std::string(e.what()) + "; ";
                    }
                }
                
                // ===== SECONDARY: Heuristic CPU/Memory Adjustments =====
                // (Simulated via queue size changes at bottleneck links)
                
                if (alloc.HasMember("india_cpu")) {
                    double cpu_ratio = alloc["india_cpu"].GetDouble();
                    
                    // Adjust India bottleneck queue size based on CPU
                    // Higher CPU → larger queues (can handle more)
                    Ptr<NetDevice> indiaBottleneck = indiaRouterToGNodeB.Get(0);
                    Ptr<QueueDisc> qdisc = indiaBottleneck->GetNode()
                        ->GetObject<TrafficControlLayer>()
                        ->GetRootQueueDiscOnDevice(indiaBottleneck);
                    
                    if (qdisc) {
                        // Scale queue: 50-200 packets based on CPU ratio
                        uint32_t base_queue = 100;
                        uint32_t new_queue = (uint32_t)(base_queue * cpu_ratio * 1.5);
                        new_queue = std::max(50u, std::min(200u, new_queue));
                        
                        qdisc->SetMaxSize(QueueSize(QueueSizeUnit::PACKETS, new_queue));
                        
                        std::cout << "  ✓ India Queue Size (CPU proxy): " 
                                 << new_queue << " packets\n";
                        details += "India queue: " + std::to_string(new_queue) + "p; ";
                    }
                }
                
                if (alloc.HasMember("uk_cpu")) {
                    double cpu_ratio = alloc["uk_cpu"].GetDouble();
                    
                    Ptr<NetDevice> ukBottleneck = ukRouterToGNodeB.Get(0);
                    Ptr<QueueDisc> qdisc = ukBottleneck->GetNode()
                        ->GetObject<TrafficControlLayer>()
                        ->GetRootQueueDiscOnDevice(ukBottleneck);
                    
                    if (qdisc) {
                        uint32_t base_queue = 100;
                        uint32_t new_queue = (uint32_t)(base_queue * cpu_ratio * 1.5);
                        new_queue = std::max(50u, std::min(200u, new_queue));
                        
                        qdisc->SetMaxSize(QueueSize(QueueSizeUnit::PACKETS, new_queue));
                        
                        std::cout << "  ✓ UK Queue Size (CPU proxy): " 
                                 << new_queue << " packets\n";
                        details += "UK queue: " + std::to_string(new_queue) + "p; ";
                    }
                }
                
                std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
                
                // Mark as successful GWO application
                if (success) {
                    details = "GWO allocation applied successfully: " + details;
                }
            } else {
                std::cerr << "⚠️  GWO action missing 'allocation' field\n";
                details += "GWO action malformed; ";
            }
        }

        // Send acknowledgment
        SendActionAck(actionId, success, details);
    }

    std::string m_aiHost;
    uint16_t m_aiPort;
    int m_socket;
    struct sockaddr_in m_aiAddr;
    bool m_socketReady;
    std::string m_lastAction;
    
    uint32_t m_messagesSent;
    uint32_t m_messagesReceived;
    uint32_t m_errors;
};

NS_LOG_COMPONENT_DEFINE("MEC5GFullSimulation");

// ============================================================================
// GLOBAL VARIABLES FOR METRICS COLLECTION
// ============================================================================
std::ofstream g_metricsFile;
Ptr<FlowMonitor> g_monitor;
FlowMonitorHelper g_flowmon;

std::map<Ptr<Socket>, FlowId> g_socketToFlow;

// ============================================================================
// GLOBAL RL SOCKET INSTANCE
// ============================================================================
RLSocketInterface* g_rlSocket = nullptr;
// Track which flows already have CWND trace attached
static std::set<FlowId> g_tracedFlowIds;

// Counters for packet tracking
uint64_t g_totalTxPackets = 0;
uint64_t g_totalRxPackets = 0;
uint64_t g_totalLostPackets = 0;

// Per-service counters
struct ServiceMetrics {
    uint64_t txPackets = 0;
    uint64_t rxPackets = 0;
    uint64_t txBytes = 0;
    uint64_t rxBytes = 0;
    double totalDelay = 0.0;
};
std::map<std::string, ServiceMetrics> g_serviceMetrics;

// ============================================================================
// METRICS COLLECTION FUNCTIONS
// ============================================================================

void PrintMetricsHeader() {
    g_metricsFile << "timestamp,service_type,flow_id,src_ip,dst_ip,"
                  << "tx_packets,rx_packets,tx_bytes,rx_bytes,lost_packets,loss_rate,"
                  << "throughput_mbps,avg_delay_ms,jitter_ms,cwnd,queue_packets" << std::endl;
}

void TraceTcpCwnd(Ptr<Application> app, FlowKey flowKey)
{
    Ptr<OnOffApplication> onoff = DynamicCast<OnOffApplication>(app);
    if (!onoff) return;

    Ptr<Socket> socket = onoff->GetSocket();
    if (socket)
    {
        // Avoid duplicate trace
        if (g_flowCwndByTuple.find(flowKey) != g_flowCwndByTuple.end()) return;

        socket->TraceConnectWithoutContext(
            "CongestionWindow",
            MakeBoundCallback(
                +[](FlowKey key, uint32_t oldCwnd, uint32_t newCwnd) {
                    g_flowCwndByTuple[key] = newCwnd;
                },
                flowKey
            )
        );

        std::cout << "✔ CWND trace connected for "
                  << flowKey.srcAddr << ":" << flowKey.srcPort
                  << " → " << flowKey.dstAddr << ":" << flowKey.dstPort << std::endl;
    }
    else
    {
        static std::map<FlowKey, int> retryCount;
        retryCount[flowKey]++;
        if (retryCount[flowKey] < 50)  // retry up to 50 times (~0.5 s)
        {
            Simulator::Schedule(MilliSeconds(10), &TraceTcpCwnd, app, flowKey);
        }
        else
        {
            std::cerr << "❌ CWND trace connection failed for "
                      << flowKey.srcAddr << ":" << flowKey.srcPort
                      << " → " << flowKey.dstAddr << ":" << flowKey.dstPort << std::endl;
        }
    }
}


// Helper function to create FlowKey from addresses
FlowKey MakeFlowKey(Ipv4Address src, uint16_t srcPort, 
                    Ipv4Address dst, uint16_t dstPort, 
                    uint8_t protocol = 6)  // 6 = TCP
{
    FlowKey key;
    key.srcAddr = src;
    key.dstAddr = dst;
    key.srcPort = srcPort;
    key.dstPort = dstPort;
    key.protocol = protocol;
    return key;
}

void CollectAndPrintMetrics() {
    double now = Simulator::Now().GetSeconds();
    
    // Define observation window
    static double lastCollectionTime = 0.0;
    double observationWindow = now - lastCollectionTime;
    if (observationWindow < 0.1) observationWindow = 2.0; // First call uses collection interval
    lastCollectionTime = now;
    
    g_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(
        g_flowmon.GetClassifier());
    
    std::map<FlowId, FlowMonitor::FlowStats> stats = g_monitor->GetFlowStats();
    
    // Console output header
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        Network Metrics at t=" << std::fixed << std::setprecision(1) 
              << std::setw(6) << now << "s                            ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    
    // Aggregate metrics per service type
    std::map<std::string, uint64_t> serviceRxBytes;  // Accumulate received bytes
    std::map<std::string, double> serviceDelay;
    std::map<std::string, int> serviceFlowCount;

    // Clear previous mappings at start of collection
    g_flowIdToPort.clear();
    
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); 
         i != stats.end(); ++i) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);  // ✅ FIXED: i->first (not FlowId)
        
        // Determine service type based on port
        std::string serviceType = "Unknown";
        uint16_t dstPort = t.destinationPort;
        uint16_t srcPort = t.sourcePort;

        // eMBB: ports 9, 5001, 8080 (video), 554 (streaming), 555, 556
        if (dstPort == 9 || dstPort == 5001 || dstPort == 8080 || dstPort == 554 || dstPort == 555 || dstPort == 556 ||
            srcPort == 9 || srcPort == 5001 || srcPort == 8080 || srcPort == 554 || srcPort ==555 || srcPort == 556 ) {
            serviceType = "eMBB";
        } 
        // URLLC: ports 10-13
        else if ((dstPort >= 10 && dstPort <= 13) || (srcPort >= 10 && srcPort <= 13)) {
            serviceType = "URLLC";
        } 
        // mMTC: ports 8000-8199
        else if ((dstPort >= 8000 && dstPort <= 8199) || (srcPort >= 8000 && srcPort <= 8199)) {
            serviceType = "mMTC";
        }
        // Background: ports 9000+
        else if ((dstPort >= 9000 && dstPort <= 9100) || (srcPort >= 9000 && srcPort <= 9100)) {
            serviceType = "Background";
        }
        // ✅ NEW: Store FlowID → Port mapping for RL actions
        if (t.protocol == 6) {  // TCP (6) or UDP (17)
            bool isServerPort = false;
            uint16_t serverPort = 0;
            
            // Check server ports
            std::set<uint16_t> trackedPorts = {5001, 9000, 9001};
            
            if (trackedPorts.find(t.sourcePort) != trackedPorts.end()) {
                serverPort = t.sourcePort;
                isServerPort = true;
            }
            else if (trackedPorts.find(t.destinationPort) != trackedPorts.end()) {
                serverPort = t.destinationPort;
                isServerPort = true;
            }
            
            if (isServerPort) {
                g_flowIdToPort[i->first] = serverPort;
                std::cout << "✓ Mapped TCP Flow " << i->first 
                         << " (port " << serverPort << ")\n";
            }
        }
        

        // Debug classification (optional - remove after verification)
        if (t.protocol == 6) {  // TCP only
            std::cout << "Flow " << i->first 
                      << " | Ports: " << t.sourcePort << "→" << t.destinationPort 
                      << " | Classified as: " << serviceType << std::endl;
        }
        // Print current mappings for debugging
        // std::cout << "\n══════ Current Flow Mappings ══════\n";
        // for (const auto& mapping : g_flowIdToPort) {
        //     std::cout << "Flow " << mapping.first << " → Port " << mapping.second << "\n";
        // }
        // std::cout << "══════════════════════════════════\n\n";

        // ✅ NEW: Get CWND using flow 5-tuple
        FlowKey key = MakeFlowKey(t.sourceAddress, t.sourcePort, 
                                   t.destinationAddress, t.destinationPort, 
                                   t.protocol);

        // --- Auto-attach CWND trace for each TCP flow (only once) ---
        if (t.protocol == 6) { // TCP only
            FlowId fid = i->first;
            
            if (g_tracedFlowIds.find(fid) == g_tracedFlowIds.end()) {
                // ✅ NEW: Try to find socket by destination port
                uint16_t destPort = t.destinationPort;
                Ptr<Socket> sockToTrace = nullptr;

                auto portIt = g_portToSocket.find(destPort);
                if (portIt != g_portToSocket.end()) {
                    sockToTrace = portIt->second;
                } else {
                    // Fallback: try reverse direction (source port)
                    portIt = g_portToSocket.find(t.sourcePort);
                    if (portIt != g_portToSocket.end()) {
                        sockToTrace = portIt->second;
                    }
                }

                if (sockToTrace != nullptr) {
                    sockToTrace->TraceConnectWithoutContext(
                        "CongestionWindow",
                        MakeBoundCallback(
                            +[](FlowKey flowKey, uint32_t oldCwnd, uint32_t newCwnd) {
                                g_flowCwndByTuple[flowKey] = newCwnd;
                            },
                            key
                        )
                    );
                    g_tracedFlowIds.insert(fid);
                    std::cout << "✔ Auto-attached CWND trace for Flow " << fid
                            << " (port " << destPort << ")\n";
                } else {
                    static std::map<FlowId, int> retryCount;
                    retryCount[fid]++;
                    if (retryCount[fid] == 1) {  // Only warn once
                        std::cerr << "⚠️  No socket found for Flow " << fid 
                                << " (port " << destPort << ")\n";
                    }
                }
            }
        }

        uint32_t cwnd = 0;
        auto it = g_flowCwndByTuple.find(key);
        if (it != g_flowCwndByTuple.end()) {
            cwnd = it->second;
        }
        
        // Calculate metrics
        double txPackets = i->second.txPackets;
        double rxPackets = i->second.rxPackets;
        double lostPackets = txPackets - rxPackets;
        double lossRate = (txPackets > 0) ? (lostPackets / txPackets) * 100.0 : 0.0;
        
        // ✅ Accumulate received bytes (not throughput)
        serviceRxBytes[serviceType] += i->second.rxBytes;
        
        // Calculate per-flow throughput for CSV
        double flowThroughput = 0.0;
        if (i->second.timeLastRxPacket.GetSeconds() > i->second.timeFirstTxPacket.GetSeconds()) {
            double flowDuration = i->second.timeLastRxPacket.GetSeconds() - 
                                i->second.timeFirstTxPacket.GetSeconds();
            flowThroughput = (i->second.rxBytes * 8.0) / (flowDuration * 1000000.0);
        }
        
        double avgDelay = 0.0;
        if (rxPackets > 0) {
            avgDelay = (i->second.delaySum.GetMilliSeconds()) / rxPackets;
        }
        
        double jitter = 0.0;
        if (rxPackets > 1) {
            jitter = (i->second.jitterSum.GetMilliSeconds()) / (rxPackets - 1);
        }
        
        // Aggregate by service type
        serviceDelay[serviceType] += avgDelay;
        serviceFlowCount[serviceType]++;
        
        // Write to CSV (per-flow metrics)
        g_metricsFile << now << ","
                     << serviceType << ","
                     << i->first << ","
                     << t.sourceAddress << ","
                     << t.destinationAddress << ","
                     << txPackets << ","
                     << rxPackets << ","
                     << i->second.txBytes << ","      // ✅ Added txBytes
                     << i->second.rxBytes << ","      // ✅ Added rxBytes
                     << lostPackets << ","
                     << std::fixed << std::setprecision(2) << lossRate << ","
                     << flowThroughput << ","
                     << avgDelay << ","
                     << jitter << ","
                     << cwnd << ","
                     << "0" << std::endl;
    }
    
    // ✅ Calculate aggregate throughput per service (CORRECTED)
    std::map<std::string, double> serviceThroughput;
    for (auto const& [service, rxBytes] : serviceRxBytes) {
        serviceThroughput[service] = (rxBytes * 8.0) / (observationWindow * 1000000.0);
    }
    
    // Print aggregated console output
    std::cout << "║ eMBB (Enhanced Mobile Broadband):                             ║\n";
    std::cout << "║   Throughput: " << std::setw(8) << std::fixed << std::setprecision(2) 
              << serviceThroughput["eMBB"] << " Mbps  |  Flows: " << std::setw(2) 
              << serviceFlowCount["eMBB"] << "                               ║\n";
    
    std::cout << "║ URLLC (Ultra-Reliable Low Latency):                           ║\n";
    double avgUrllcDelay = (serviceFlowCount["URLLC"] > 0) ? 
                           serviceDelay["URLLC"] / serviceFlowCount["URLLC"] : 0.0;
    std::cout << "║   Avg Delay: " << std::setw(8) << avgUrllcDelay << " ms    |  Flows: " 
              << std::setw(2) << serviceFlowCount["URLLC"] << "                    ║\n";
    
    std::cout << "║ mMTC (Massive Machine-Type Comms):                             ║\n";
    std::cout << "║   Active Flows: " << std::setw(2) << serviceFlowCount["mMTC"] 
              << "                                                          ║\n";
     
    std::cout << "║ Background Traffic:                                            ║\n";
    std::cout << "║   Throughput: " << std::setw(8) << std::fixed << std::setprecision(2) 
              << serviceThroughput["Background"] << " Mbps  |  Flows: " << std::setw(2) 
              << serviceFlowCount["Background"] << "                            ║\n";

    std::cout << "║ Unknown/Unclassified:                                          ║\n";
    std::cout << "║   Flows: " << std::setw(2) << serviceFlowCount["Unknown"] 
              << "                                                              ║\n";              
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        
    // ============================================================================
    // SEND METRICS TO AI CONTROLLER VIA SOCKET
    // ============================================================================
    if (g_rlSocket && g_rlSocket->IsReady()) {
        std::ostringstream metricsJson;
        metricsJson << std::fixed << std::setprecision(4);
        metricsJson << "{"
                    << "\"type\":\"state\","
                    << "\"timestamp\":" << now << ","
                    << "\"flows\":[";
        
        bool firstFlow = true;
        for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); 
             i != stats.end(); ++i) {
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
            
            // Determine service type (same logic as before)
            std::string serviceType = "Unknown";
            uint16_t dstPort = t.destinationPort;
            uint16_t srcPort = t.sourcePort;

            if (dstPort == 9 || dstPort == 5001 || dstPort == 8080 || dstPort == 554 || dstPort == 555 || dstPort == 556 ||
                srcPort == 9 || srcPort == 5001 || srcPort == 8080 || srcPort == 554 || srcPort ==555 || srcPort == 556 ) {
                serviceType = "eMBB";
            } 
            else if ((dstPort >= 10 && dstPort <= 13) || (srcPort >= 10 && srcPort <= 13)) {
                serviceType = "URLLC";
            } 
            else if ((dstPort >= 8000 && dstPort <= 8199) || (srcPort >= 8000 && srcPort <= 8199)) {
                serviceType = "mMTC";
            }
            else if ((dstPort >= 9000 && dstPort <= 9100) || (srcPort >= 9000 && srcPort <= 9100)) {
                serviceType = "Background";
            }
            
            // Calculate metrics
            double txPackets = i->second.txPackets;
            double rxPackets = i->second.rxPackets;
            double lostPackets = txPackets - rxPackets;
            double lossRate = (txPackets > 0) ? (lostPackets / txPackets) * 100.0 : 0.0;
            
            double throughput = 0.0;
            if (i->second.timeLastRxPacket.GetSeconds() > i->second.timeFirstTxPacket.GetSeconds()) {
                double duration = i->second.timeLastRxPacket.GetSeconds() - 
                                i->second.timeFirstTxPacket.GetSeconds();
                throughput = (i->second.rxBytes * 8.0) / (duration * 1000000.0);
            }
            
            double avgDelay = 0.0;
            if (rxPackets > 0) {
                avgDelay = (i->second.delaySum.GetMilliSeconds()) / rxPackets;
            }
            
            double jitter = 0.0;
            if (rxPackets > 1) {
                jitter = (i->second.jitterSum.GetMilliSeconds()) / (rxPackets - 1);
            }
            
            uint32_t cwnd = 0;
            FlowKey key = MakeFlowKey(t.sourceAddress, t.sourcePort,
                           t.destinationAddress, t.destinationPort,
                           t.protocol);

            auto cwndIt = g_flowCwndByTuple.find(key);
            if (cwndIt != g_flowCwndByTuple.end()) {
                cwnd = cwndIt->second;
            } else {
                cwnd = 0;
            }

            
            // Build JSON for this flow
            if (!firstFlow) metricsJson << ",";
            firstFlow = false;
            
            metricsJson << "{"
                       << "\"id\":" << i->first << ","
                       << "\"service_type\":\"" << serviceType << "\","
                       << "\"src\":\"" << t.sourceAddress << "\","
                       << "\"dst\":\"" << t.destinationAddress << "\","
                       << "\"src_port\":" << t.sourcePort << ","
                       << "\"dst_port\":" << t.destinationPort << ","
                       << "\"protocol\":\"" << (t.protocol == 6 ? "TCP" : "UDP") << "\"," // Changed to string
                       << "\"protocol_num\":" << (int)t.protocol << "," // Keep numeric version too
                       << "\"tx_packets\":" << (uint64_t)txPackets << ","
                       << "\"rx_packets\":" << (uint64_t)rxPackets << ","
                       << "\"tx_bytes\":" << i->second.txBytes << ","
                       << "\"rx_bytes\":" << i->second.rxBytes << ","
                       << "\"lost_packets\":" << (uint64_t)lostPackets << ","
                       << "\"loss_rate\":" << lossRate << ","
                       << "\"throughput_mbps\":" << throughput << ","
                       << "\"rtt_ms\":" << avgDelay << ","
                       << "\"jitter_ms\":" << jitter << ","
                       << "\"cwnd\":" << cwnd
                       << "}";
        }
        
        metricsJson << "],"
                    << "\"queues\":[";
        
        // TODO: Add queue statistics if available
        metricsJson << "{\"device\":\"india_bottleneck\",\"length\":0,\"drops\":0},"
                    << "{\"device\":\"uk_bottleneck\",\"length\":0,\"drops\":0}";
        
        metricsJson << "],"
                    << "\"edges\":["
                    << "{\"location\":\"india\",\"cpu_usage\":0.0,\"memory_usage\":0.0},"
                    << "{\"location\":\"uk\",\"cpu_usage\":0.0,\"memory_usage\":0.0}"
                    << "]"
                    << "}";
        
        // Send to AI controller
        g_rlSocket->SendMetrics(metricsJson.str());
    }

    // Schedule next metrics collection
    Simulator::Schedule(Seconds(2.0), &CollectAndPrintMetrics);
}
// ============================================================================
// HELPER FUNCTION TO REGISTER FLOW-TO-APPLICATION MAPPINGS
// ============================================================================
/**
 * Register a TCP OnOff application for RL control
 * Call this after installing OnOff applications
 */
void RegisterTcpFlow(ApplicationContainer& apps, uint16_t destPort, const std::string& flowName)
{
    if (apps.GetN() == 0) {
        std::cerr << "⚠️  Warning: Empty application container for " << flowName << std::endl;
        return;
    }
    
    Ptr<OnOffApplication> onoff = DynamicCast<OnOffApplication>(apps.Get(0));
    if (onoff) {
        g_portToOnOffApp[destPort] = onoff;
        
        Ptr<Socket> socket = onoff->GetSocket();
        if (socket) {
            g_portToSocket[destPort] = socket;
            
            std::cout << "✓ Registered TCP Flow to port " << destPort 
                      << " (" << flowName << ")\n";
        } else {
            // Socket not created yet - schedule retry
            Simulator::Schedule(MilliSeconds(100), [apps, destPort, flowName]() {
                RegisterTcpFlow(const_cast<ApplicationContainer&>(apps), destPort, flowName);
            });
        }
    } else {
        std::cerr << "⚠️  Failed to cast application for " << flowName << std::endl;
    }
}

/**
 * Register a UDP Echo client for RL control
 */
void RegisterUdpFlow(ApplicationContainer& apps, uint16_t destPort, const std::string& flowName)
{
    if (apps.GetN() == 0) {
        std::cerr << "⚠️  Warning: Empty application container for " << flowName << std::endl;
        return;
    }
    
    // For UDP Echo applications, we can't get socket directly
    // Just log and skip registration (UDP flows don't need RL control as much)
    Ptr<UdpEchoClient> echoClient = DynamicCast<UdpEchoClient>(apps.Get(0));
    if (echoClient) {
        std::cout << "✓ Registered UDP Echo Flow to port " << destPort 
                  << " (" << flowName << ") - No socket control needed\n";
        return;
    }
    
    // Try OnOffApplication (for UDP OnOff apps)
    Ptr<OnOffApplication> onoff = DynamicCast<OnOffApplication>(apps.Get(0));
    if (onoff) {
        Ptr<Socket> socket = onoff->GetSocket();
        if (socket) {
            g_portToSocket[destPort] = socket;
            g_portToOnOffApp[destPort] = onoff;
            std::cout << "✓ Registered UDP OnOff Flow to port " << destPort 
                      << " (" << flowName << ")\n";
            return;
        } else {
            // Socket not created yet - schedule retry
            Simulator::Schedule(MilliSeconds(100), [apps, destPort, flowName]() {
                RegisterUdpFlow(const_cast<ApplicationContainer&>(apps), destPort, flowName);
            });
            return;
        }
    }
    
    // If we get here, it's an application type we don't handle
    std::cout << "ℹ️  UDP Flow to port " << destPort << " (" << flowName 
              << ") - Application type not registered for RL control\n";
}

// ============================================================================
// ADMISSION CONTROL GLOBALS AND FUNCTIONS (PUT BEFORE main())
// ============================================================================

// Global counters for admission control
uint32_t g_activeURLLCFlows = 0;
const uint32_t MAX_URLLC_FLOWS = 10; // Maximum concurrent URLLC flows

// Helper function to check if socket is URLLC traffic
bool IsURLLC(Ptr<Socket> socket) {
    // Check by destination port (URLLC ports: 10-13)
    Address peerAddress;
    if (socket->GetPeerName(peerAddress)) {
        InetSocketAddress inetAddr = InetSocketAddress::ConvertFrom(peerAddress);
        uint16_t port = inetAddr.GetPort();
        return (port >= 10 && port <= 13);
    }
    return false;
}

// Connection request handler for admission control - FIXED SIGNATURE
bool ShouldAcceptConnection(Ptr<Socket> socket, const Address& address) {
    if (IsURLLC(socket)) {
        if (g_activeURLLCFlows >= MAX_URLLC_FLOWS) {
            // Reject URLLC flow due to overload
            NS_LOG_INFO("URLLC flow rejected - maximum capacity reached (" 
                    << MAX_URLLC_FLOWS << " flows active)");
            std::cout << "⚠️ URLLC flow rejected - at capacity\n";
            return false; // Reject connection
        }
        g_activeURLLCFlows++;
        NS_LOG_INFO("URLLC flow accepted (" << g_activeURLLCFlows << "/" 
                << MAX_URLLC_FLOWS << " active)");
    }
    
    return true; // Accept connection
}

// Normal connection callback - FIXED SIGNATURE  
void NormalAcceptCallback(Ptr<Socket> socket, const Address& address) {
    // Connection was accepted, nothing special to do
    NS_LOG_DEBUG("Connection accepted from " << address);
}

// Connection close handler to update counters - FIXED SIGNATURE
void OnNormalClose(Ptr<Socket> socket) {
    NS_LOG_DEBUG("Connection closed normally");
    if (IsURLLC(socket)) {
        if (g_activeURLLCFlows > 0) {
            g_activeURLLCFlows--;
            NS_LOG_INFO("URLLC flow closed (" << g_activeURLLCFlows << "/" 
                    << MAX_URLLC_FLOWS << " active)");
        }
    }
}

// Error close handler - FIXED SIGNATURE
void OnErrorClose(Ptr<Socket> socket) {
    NS_LOG_DEBUG("Connection closed due to error");
    if (IsURLLC(socket)) {
        if (g_activeURLLCFlows > 0) {
            g_activeURLLCFlows--;
            NS_LOG_INFO("URLLC flow error closed (" << g_activeURLLCFlows << "/" 
                    << MAX_URLLC_FLOWS << " active)");
        }
    }
}

// Install admission control on gNodeB TCP sockets
void InstallAdmissionControl(Ptr<Node> gNodeB) {
    // Create listening socket for incoming connections
    Ptr<Socket> listenSocket = Socket::CreateSocket(gNodeB, TcpSocketFactory::GetTypeId());
    
    // Bind to all interfaces (port 0 = any available port)
    listenSocket->Bind();
    listenSocket->Listen();
    
    // Set connection handlers with CORRECT SIGNATURES
    listenSocket->SetAcceptCallback(
        MakeCallback(&ShouldAcceptConnection),    // bool (Ptr<Socket>, const Address&)
        MakeCallback(&NormalAcceptCallback)       // void (Ptr<Socket>, const Address&)
    );
    
    // Set close callbacks with CORRECT SIGNATURES
    listenSocket->SetCloseCallbacks(
        MakeCallback(&OnNormalClose),             // void (Ptr<Socket>)
        MakeCallback(&OnErrorClose)               // void (Ptr<Socket>)
    );
    
    NS_LOG_INFO("Admission control installed on gNodeB");
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char *argv[])
{
    // Load config if provided
    int numEmbbFlows = 2;  // defaults
    int numUrllcFlows = 2;
    int numMmtcDevices = 10;
    int numBackgroundFlows = 2;
    double embbRateMin = 30.0;
    double embbRateMax = 60.0;

    double simTime = 60.0;
    bool verbose = false;
    std::string metricsFile = "mec_metrics.csv";

    // RL Socket configuration
    bool enableRL = false;
    std::string aiHost = "127.0.0.1";
    uint16_t aiPort = 5000;

    // TCP Congestion Control Algorithm Selection
    std::string tcpAlgo = "TcpBbr";  // Default: BBR
    std::string configFile = "";
    CommandLine cmd;
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("verbose", "Enable verbose logging", verbose);
    cmd.AddValue("metricsFile", "CSV file for metrics output", metricsFile);
    cmd.AddValue("enableRL", "Enable RL socket communication", enableRL);
    cmd.AddValue("aiHost", "AI controller host address", aiHost);
    cmd.AddValue("aiPort", "AI controller port", aiPort);
    cmd.AddValue("tcpAlgo", "TCP algorithm: TcpBbr, TcpCubic, TcpVegas, TcpNewReno, TcpHighSpeed, TcpHybla, TcpWestwood, TcpVeno", tcpAlgo);
    cmd.AddValue("config", "JSON config file for traffic generation", configFile);
    cmd.Parse(argc, argv);

    // Load config if provided
    if (!configFile.empty()) {
        std::cout << "📄 Loading config from: " << configFile << std::endl;
        
        FILE* fp = fopen(configFile.c_str(), "r");
        if (fp) {
            char readBuffer[65536];
            rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
            
            rapidjson::Document configDoc;
            configDoc.ParseStream(is);
            fclose(fp);
            
            if (!configDoc.HasParseError()) {
                // Extract traffic parameters
                if (configDoc.HasMember("num_embb_flows")) {
                    numEmbbFlows = configDoc["num_embb_flows"].GetInt();
                }
                if (configDoc.HasMember("num_urllc_flows")) {
                    numUrllcFlows = configDoc["num_urllc_flows"].GetInt();
                }
                if (configDoc.HasMember("num_mmtc_devices")) {
                    numMmtcDevices = configDoc["num_mmtc_devices"].GetInt();
                }
                if (configDoc.HasMember("num_background_flows")) {
                    numBackgroundFlows = configDoc["num_background_flows"].GetInt();
                }
                if (configDoc.HasMember("embb_rate_min")) {
                    embbRateMin = configDoc["embb_rate_min"].GetDouble();
                }
                if (configDoc.HasMember("embb_rate_max")) {
                    embbRateMax = configDoc["embb_rate_max"].GetDouble();
                }
                
                std::cout << "✓ Config loaded: "
                        << "eMBB=" << numEmbbFlows
                        << ", URLLC=" << numUrllcFlows
                        << ", mMTC=" << numMmtcDevices << std::endl;
            }
        }
    }

    if (enableRL) {
        std::cout << "\n🤖 RL MODE ENABLED\n";
        std::cout << "   AI Controller: " << aiHost << ":" << aiPort << "\n";
        std::cout << "   ⚠️  Make sure Python AI controller is running!\n\n";
        
        g_rlSocket = new RLSocketInterface(aiHost, aiPort);
        
        if (!g_rlSocket->Initialize()) {
            std::cout << "❌ Failed to initialize RL socket!\n";
            std::cout << "   Continuing without RL integration...\n\n";
            delete g_rlSocket;
            g_rlSocket = nullptr;
        } else {
            std::cout << "✅ RL socket initialized successfully\n\n";
        }
    } else {
        std::cout << "\n📊 BASELINE MODE (No RL)\n";
        std::cout << "   Run with --enableRL=true to enable AI control\n\n";
    }

    // ============================================================================
    // RANDOMIZATION FOR RL TRAINING (DIFFERENT PATTERN EVERY RUN)
    // ============================================================================
    uint32_t runSeed = (uint32_t)time(nullptr);  // Use system time as seed
    RngSeedManager::SetSeed(runSeed);
    RngSeedManager::SetRun(1);

    // ============================================================================
    // INITIALIZE RL SOCKET (OPTIONAL - CONTROLLED BY COMMAND LINE)
    // ============================================================================
    // bool enableRL = false;
    // std::string aiHost = "127.0.0.1";
    // uint16_t aiPort = 5000;
    
    cmd.AddValue("enableRL", "Enable RL socket communication", enableRL);
    cmd.AddValue("aiHost", "AI controller host address", aiHost);
    cmd.AddValue("aiPort", "AI controller port", aiPort);

    std::cout << "\n🎲 RANDOMIZATION SEED: " << runSeed << " (save for reproducibility)\n";
    std::cout << "   To reproduce this run: RngSeedManager::SetSeed(" << runSeed << ")\n\n";
 
    // Random number generators for traffic diversity
    Ptr<UniformRandomVariable> startTimeRand = CreateObject<UniformRandomVariable>();
    startTimeRand->SetAttribute("Min", DoubleValue(2.0));
    startTimeRand->SetAttribute("Max", DoubleValue(6.0));  // Flows start 2-6s (not always at 3s!)

    Ptr<UniformRandomVariable> dataRateRand = CreateObject<UniformRandomVariable>();
    // Will set Min/Max per service type

    Ptr<UniformRandomVariable> iotCountRand = CreateObject<UniformRandomVariable>();
    iotCountRand->SetAttribute("Min", DoubleValue(5.0));
    iotCountRand->SetAttribute("Max", DoubleValue(20.0));  // 5-20 IoT devices per run

    Ptr<UniformRandomVariable> packetSizeRand = CreateObject<UniformRandomVariable>();
    // For realistic packet size variation

    Time::SetResolution(Time::NS);

    // ============================================================================
    // TCP CONGESTION CONTROL ALGORITHM SELECTION (DYNAMIC)
    // ============================================================================
    
    // Validate and set TCP algorithm
    std::map<std::string, std::string> tcpAlgoMap = {
        {"TcpBbr", "ns3::TcpBbr"},
        {"TcpCubic", "ns3::TcpCubic"},
        {"TcpVegas", "ns3::TcpVegas"},
        {"TcpNewReno", "ns3::TcpNewReno"},
        {"TcpHighSpeed", "ns3::TcpHighSpeed"},
        {"TcpHybla", "ns3::TcpHybla"},
        {"TcpWestwood", "ns3::TcpWestwood"},
        {"TcpVeno", "ns3::TcpVeno"}
    };
    
    std::string tcpFullName;
    if (tcpAlgoMap.find(tcpAlgo) != tcpAlgoMap.end()) {
        tcpFullName = tcpAlgoMap[tcpAlgo];
    } else {
        std::cerr << "\n❌ ERROR: Invalid TCP algorithm '" << tcpAlgo << "'\n";
        std::cerr << "   Available options:\n";
        for (const auto& [key, value] : tcpAlgoMap) {
            std::cerr << "     - " << key << "\n";
        }
        std::cerr << "\n   Defaulting to TcpBbr\n\n";
        tcpAlgo = "TcpBbr";
        tcpFullName = "ns3::TcpBbr";
    }
    
    // Apply TCP algorithm (overridden if RL mode with full control)
    if (enableRL) {
        std::cout << "\n🤖 RL MODE: Using TCP " << tcpAlgo << " as baseline\n";
        std::cout << "   (RL agent will observe/modify behavior)\n";
    } else {
        std::cout << "\n📊 BASELINE MODE: Using TCP " << tcpAlgo << "\n";
    }
    
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", StringValue(tcpFullName));
    
    // Optional: Set initial congestion window (matches typical Linux behavior)
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1400));

    std::cout << "   Algorithm: " << tcpAlgo << "\n";
    std::cout << "   Initial CWND: 10 segments\n";
    std::cout << "   Segment Size: 1400 bytes\n\n";

    if (verbose) {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
        LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
        LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
    }

    // Open metrics file
    g_metricsFile.open(metricsFile);
    PrintMetricsHeader();

    NS_LOG_INFO("Creating 5G MEC Topology with Diverse Traffic Patterns");
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     5G MEC NETWORK SIMULATION WITH CONGESTION SCENARIOS      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // ============================================================================
    // TOPOLOGY SETUP (KEEPING YOUR STRUCTURE INTACT)
    // ============================================================================
    // NetDeviceContainer indiaRouterToGNodeB, indiaGNodeBToQoS, indiaQoSToMEC;
    // NetDeviceContainer indiaMECToUPF, indiaUPFToBackbone;
    // NetDeviceContainer ukRouterToGNodeB, ukGNodeBToQoS, ukQoSToMEC;
    // NetDeviceContainer ukMECToUPF, ukUPFToBackbone;
    // NetDeviceContainer backboneLink, backboneToCloud, backboneToInternet;

    // Node creation
    NodeContainer indiaLAN_eMBB, indiaLAN_URLLC, indiaLAN_mMTC;
    indiaLAN_eMBB.Create(1);
    indiaLAN_URLLC.Create(1);
    indiaLAN_mMTC.Create(1);

    NodeContainer indiaRouterNode, indiaGNodeB, indiaQoSNode, indiaMECApps, indiaUPF;
    indiaRouterNode.Create(1);
    indiaGNodeB.Create(1);
    indiaQoSNode.Create(1);
    indiaMECApps.Create(1);
    indiaUPF.Create(1);

    NodeContainer ukLAN_eMBB, ukLAN_URLLC, ukLAN_mMTC;
    ukLAN_eMBB.Create(1);
    ukLAN_URLLC.Create(1);
    ukLAN_mMTC.Create(1);

    NodeContainer ukRouterNode, ukGNodeB, ukQoSNode, ukMECApps, ukUPF;
    ukRouterNode.Create(1);
    ukGNodeB.Create(1);
    ukQoSNode.Create(1);
    ukMECApps.Create(1);
    ukUPF.Create(1);

    NodeContainer internationalBackbone, cloudServices, publicInternet;
    internationalBackbone.Create(2);
    cloudServices.Create(1);
    publicInternet.Create(1);

    // Link configurations
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
    csma.SetChannelAttribute("Delay", TimeValue(NanoSeconds(6560)));

    PointToPointHelper p2p;

    // LAN segments
    NodeContainer indiaLANSegment;
    indiaLANSegment.Add(indiaLAN_eMBB.Get(0));
    indiaLANSegment.Add(indiaLAN_URLLC.Get(0));
    indiaLANSegment.Add(indiaLAN_mMTC.Get(0));
    indiaLANSegment.Add(indiaRouterNode.Get(0));
    NetDeviceContainer indiaLANDevices = csma.Install(indiaLANSegment);

    NodeContainer ukLANSegment;
    ukLANSegment.Add(ukLAN_eMBB.Get(0));
    ukLANSegment.Add(ukLAN_URLLC.Get(0));
    ukLANSegment.Add(ukLAN_mMTC.Get(0));
    ukLANSegment.Add(ukRouterNode.Get(0));
    NetDeviceContainer ukLANDevices = csma.Install(ukLANSegment);

    // India RAN chain
    p2p.SetDeviceAttribute("DataRate", StringValue("200Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    indiaRouterToGNodeB = p2p.Install(NodeContainer(indiaRouterNode.Get(0), indiaGNodeB.Get(0)));



    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));
    indiaGNodeBToQoS = p2p.Install(NodeContainer(indiaGNodeB.Get(0), indiaQoSNode.Get(0)));

    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("0.5ms"));
    indiaQoSToMEC = p2p.Install(NodeContainer(indiaQoSNode.Get(0), indiaMECApps.Get(0)));

    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));
    indiaMECToUPF = p2p.Install(NodeContainer(indiaMECApps.Get(0), indiaUPF.Get(0)));

    // UK RAN chain
    p2p.SetDeviceAttribute("DataRate", StringValue("200Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    ukRouterToGNodeB = p2p.Install(NodeContainer(ukRouterNode.Get(0), ukGNodeB.Get(0)));

    

    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));
    ukGNodeBToQoS = p2p.Install(NodeContainer(ukGNodeB.Get(0), ukQoSNode.Get(0)));

    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("0.5ms"));
    ukQoSToMEC = p2p.Install(NodeContainer(ukQoSNode.Get(0), ukMECApps.Get(0)));

    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("1ms"));
    ukMECToUPF = p2p.Install(NodeContainer(ukMECApps.Get(0), ukUPF.Get(0)));

    // Backbone connections
    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("5ms"));
    indiaUPFToBackbone = p2p.Install(NodeContainer(indiaUPF.Get(0), internationalBackbone.Get(0)));
    ukUPFToBackbone = p2p.Install(NodeContainer(ukUPF.Get(0), internationalBackbone.Get(1)));

    p2p.SetChannelAttribute("Delay", StringValue("50ms"));
    backboneLink = p2p.Install(internationalBackbone);

    p2p.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("10ms"));
    backboneToCloud = p2p.Install(NodeContainer(internationalBackbone.Get(0), cloudServices.Get(0)));

    p2p.SetDeviceAttribute("DataRate", StringValue("1Gbps"));
    p2p.SetChannelAttribute("Delay", StringValue("15ms"));
    backboneToInternet = p2p.Install(NodeContainer(internationalBackbone.Get(0), publicInternet.Get(0)));

    // Internet stack
    InternetStackHelper stack;
    stack.InstallAll();

    // ============================================================================
    // ADMISSION CONTROL AT GNODEB (INSIDE main())
    // ============================================================================
    std::cout << "\n🛑 Installing Admission Control at gNodeBs...\n";

    // Install admission control on both gNodeBs
    InstallAdmissionControl(indiaGNodeB.Get(0));
    InstallAdmissionControl(ukGNodeB.Get(0));

    std::cout << "   ✅ India gNodeB: Admission Control (max " << MAX_URLLC_FLOWS << " URLLC flows)\n";
    std::cout << "   ✅ UK gNodeB: Admission Control (max " << MAX_URLLC_FLOWS << " URLLC flows)\n";
    std::cout << "   🎯 URLLC flows will be rejected during overload conditions\n";


    // ============================================================================
    // INSTALL QUEUE DISCIPLINES (AFTER INTERNET STACK!)
    // ============================================================================
    std::cout << "\n🔧 Installing Active Queue Management (AQM)...\n";

    // Use PfifoFast for both India and UK for consistent QoS
    TrafficControlHelper tchIndia;
    tchIndia.SetRootQueueDisc("ns3::PfifoFastQueueDisc");
    QueueDiscContainer indiaQdisc = tchIndia.Install(indiaRouterToGNodeB);
    std::cout << "   ✅ India bottleneck: PfifoFast (3 priority bands)\n";

    TrafficControlHelper tchUK;
    tchUK.SetRootQueueDisc("ns3::PfifoFastQueueDisc");
    QueueDiscContainer ukQdisc = tchUK.Install(ukRouterToGNodeB);
    std::cout << "   ✅ UK bottleneck: PfifoFast (3 priority bands)\n";

    // Optional: Apply to backbone link too (CoDel is fine here since it's high capacity)
    TrafficControlHelper tchBackbone;
    tchBackbone.SetRootQueueDisc("ns3::CoDelQueueDisc",
                                "MaxSize", StringValue("200p"));
    tchBackbone.Install(backboneLink);
    std::cout << "   ✅ Backbone link: CoDel (200 packet buffer)\n\n";

    // IP addressing
    Ipv4AddressHelper ipv4;

    ipv4.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer indiaLANInterfaces = ipv4.Assign(indiaLANDevices);

    ipv4.SetBase("20.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer ukLANInterfaces = ipv4.Assign(ukLANDevices);

    ipv4.SetBase("10.0.2.0", "255.255.255.252");
    Ipv4InterfaceContainer indiaRouterGNodeBInterfaces = ipv4.Assign(indiaRouterToGNodeB);

    ipv4.SetBase("20.0.2.0", "255.255.255.252");
    Ipv4InterfaceContainer ukRouterGNodeBInterfaces = ipv4.Assign(ukRouterToGNodeB);

    ipv4.SetBase("10.0.1.0", "255.255.255.0");
    Ipv4InterfaceContainer indiaGNodeBQoSInterfaces = ipv4.Assign(indiaGNodeBToQoS);
    Ipv4InterfaceContainer indiaQoSMECInterfaces = ipv4.Assign(indiaQoSToMEC);
    Ipv4InterfaceContainer indiaMECUPFInterfaces = ipv4.Assign(indiaMECToUPF);

    ipv4.SetBase("20.0.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ukGNodeBQoSInterfaces = ipv4.Assign(ukGNodeBToQoS);
    Ipv4InterfaceContainer ukQoSMECInterfaces = ipv4.Assign(ukQoSToMEC);
    Ipv4InterfaceContainer ukMECUPFInterfaces = ipv4.Assign(ukMECToUPF);

    ipv4.SetBase("172.16.0.0", "255.255.255.252");
    Ipv4InterfaceContainer indiaUPFBackboneInterfaces = ipv4.Assign(indiaUPFToBackbone);

    ipv4.SetBase("172.16.0.4", "255.255.255.252");
    Ipv4InterfaceContainer ukUPFBackboneInterfaces = ipv4.Assign(ukUPFToBackbone);

    ipv4.SetBase("172.16.0.8", "255.255.255.252");
    Ipv4InterfaceContainer backboneInterfaces = ipv4.Assign(backboneLink);

    ipv4.SetBase("203.0.113.0", "255.255.255.252");
    Ipv4InterfaceContainer backboneCloudInterfaces = ipv4.Assign(backboneToCloud);

    ipv4.SetBase("198.51.100.0", "255.255.255.252");
    Ipv4InterfaceContainer backboneInternetInterfaces = ipv4.Assign(backboneToInternet);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    NS_LOG_UNCOND("✔ Global routing tables populated successfully.");

    // Static routing (keeping your exact routing logic)
    Ipv4StaticRoutingHelper staticRoutingHelper;

    for (uint32_t i = 0; i < 3; ++i) {
        Ptr<Ipv4> ipv4Host;
        if (i == 0) ipv4Host = indiaLAN_eMBB.Get(0)->GetObject<Ipv4>();
        else if (i == 1) ipv4Host = indiaLAN_URLLC.Get(0)->GetObject<Ipv4>();
        else ipv4Host = indiaLAN_mMTC.Get(0)->GetObject<Ipv4>();
        Ptr<Ipv4StaticRouting> staticRouting = staticRoutingHelper.GetStaticRouting(ipv4Host);
        staticRouting->SetDefaultRoute(indiaLANInterfaces.GetAddress(3), 1);
    }

    for (uint32_t i = 0; i < 3; ++i) {
        Ptr<Ipv4> ipv4Host;
        if (i == 0) ipv4Host = ukLAN_eMBB.Get(0)->GetObject<Ipv4>();
        else if (i == 1) ipv4Host = ukLAN_URLLC.Get(0)->GetObject<Ipv4>();
        else ipv4Host = ukLAN_mMTC.Get(0)->GetObject<Ipv4>();
        Ptr<Ipv4StaticRouting> staticRouting = staticRoutingHelper.GetStaticRouting(ipv4Host);
        staticRouting->SetDefaultRoute(ukLANInterfaces.GetAddress(3), 1);
    }

    // 3. Optional: print routing tables to verify correctness
    Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper>("routing-tables.txt", std::ios::out);
    Ipv4GlobalRoutingHelper grHelper;
    grHelper.PrintRoutingTableAllAt(Seconds(2.0), routingStream);


    // ============================================================================
    // PRINT ASSIGNED IP ADDRESSES
    // ============================================================================
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              NETWORK IP ADDRESS ASSIGNMENTS                    ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";

    std::cout << "\n🇮🇳 INDIA LAN (10.0.0.0/24):\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  eMBB Host:   " << indiaLANInterfaces.GetAddress(0) << "\n";
    std::cout << "  URLLC Host:  " << indiaLANInterfaces.GetAddress(1) << "\n";
    std::cout << "  mMTC Host:   " << indiaLANInterfaces.GetAddress(2) << "\n";
    std::cout << "  Router:      " << indiaLANInterfaces.GetAddress(3) << "\n";

    std::cout << "\n🇬🇧 UK LAN (20.0.0.0/24):\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  eMBB Host:   " << ukLANInterfaces.GetAddress(0) << "\n";
    std::cout << "  URLLC Host:  " << ukLANInterfaces.GetAddress(1) << "\n";
    std::cout << "  mMTC Host:   " << ukLANInterfaces.GetAddress(2) << "\n";
    std::cout << "  Router:      " << ukLANInterfaces.GetAddress(3) << "\n";

    std::cout << "\n🔗 INDIA RAN + MEC (10.0.1.0/24):\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  gNodeB:      " << indiaGNodeBQoSInterfaces.GetAddress(0) << "\n";
    std::cout << "  QoS:         " << indiaQoSMECInterfaces.GetAddress(0) << "\n";
    std::cout << "  MEC:         " << indiaMECUPFInterfaces.GetAddress(0) << "\n";
    std::cout << "  UPF:         " << indiaMECUPFInterfaces.GetAddress(1) << "\n";

    std::cout << "\n🔗 UK RAN + MEC (20.0.1.0/24):\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  gNodeB:      " << ukGNodeBQoSInterfaces.GetAddress(0) << "\n";
    std::cout << "  QoS:         " << ukQoSMECInterfaces.GetAddress(0) << "\n";
    std::cout << "  MEC:         " << ukMECUPFInterfaces.GetAddress(0) << "\n";
    std::cout << "  UPF:         " << ukMECUPFInterfaces.GetAddress(1) << "\n";

    std::cout << "\n🌐 BACKBONE (172.16.0.0/16):\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  India UPF:   " << indiaUPFBackboneInterfaces.GetAddress(0) << "\n";
    std::cout << "  India BB:    " << backboneInterfaces.GetAddress(0) << "\n";
    std::cout << "  UK BB:       " << backboneInterfaces.GetAddress(1) << "\n";
    std::cout << "  UK UPF:      " << ukUPFBackboneInterfaces.GetAddress(0) << "\n";

    std::cout << "\n☁️ EXTERNAL SERVICES:\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Cloud:       " << backboneCloudInterfaces.GetAddress(1) << "\n";
    std::cout << "  Internet:    " << backboneInternetInterfaces.GetAddress(1) << "\n";

    std::cout << "\n╚════════════════════════════════════════════════════════════════╝\n\n";

    // ============================================================================
    // TRAFFIC GENERATION - DIVERSE SCENARIOS FOR CONGESTION (PROPERLY RANDOMIZED)
    // ============================================================================

    std::cout << "Setting up traffic patterns...\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // ✅ CRITICAL: Use ns-3 RNG for ALL randomization (not rand()!)
    Ptr<UniformRandomVariable> intervalRand = CreateObject<UniformRandomVariable>();

    // ========== eMBB TRAFFIC (High Bandwidth) ==========
    std::cout << "📱 eMBB Traffic (Enhanced Mobile Broadband):\n";

    // 1. India eMBB → UK eMBB (UDP Echo - FIXED)
    uint16_t port = 9;
    UdpEchoServerHelper echoServer1(port);
    ApplicationContainer serverApps1 = echoServer1.Install(ukLAN_eMBB.Get(0));
    serverApps1.Start(Seconds(1.0));
    serverApps1.Stop(Seconds(simTime));

    double embbStartTime = startTimeRand->GetValue();
    intervalRand->SetAttribute("Min", DoubleValue(70.0));
    intervalRand->SetAttribute("Max", DoubleValue(130.0));
    uint32_t embbInterval = (uint32_t)intervalRand->GetValue();

    packetSizeRand->SetAttribute("Min", DoubleValue(1100.0));
    packetSizeRand->SetAttribute("Max", DoubleValue(1700.0));
    uint32_t embbPacketSize = (uint32_t)packetSizeRand->GetValue();

    UdpEchoClientHelper echoClient1(ukLANInterfaces.GetAddress(0), port);
    echoClient1.SetAttribute("MaxPackets", UintegerValue(1000));
    echoClient1.SetAttribute("Interval", TimeValue(MilliSeconds(embbInterval)));
    echoClient1.SetAttribute("PacketSize", UintegerValue(embbPacketSize));
    echoClient1.SetAttribute("Tos", UintegerValue(0x88));  // ✅ eMBB TOS

    ApplicationContainer clientApps1 = echoClient1.Install(indiaLAN_eMBB.Get(0));
    clientApps1.Start(Seconds(embbStartTime));
    clientApps1.Stop(Seconds(simTime));

    std::cout << "  ✓ India eMBB → UK eMBB (UDP, start=" << embbStartTime 
            << "s, interval=" << embbInterval << "ms, pkt=" << embbPacketSize << "B)\n";

    // 2. UK eMBB → India eMBB (TCP Bulk - RANDOMIZED DATA RATE + DISTRIBUTION)
    uint16_t sinkPort1 = 5001;
    Address sinkAddr1(InetSocketAddress(indiaLANInterfaces.GetAddress(0), sinkPort1));
    PacketSinkHelper sinkHelper1("ns3::TcpSocketFactory", sinkAddr1);
    ApplicationContainer sinkApp1 = sinkHelper1.Install(indiaLAN_eMBB.Get(0));
    sinkApp1.Start(Seconds(1.0));
    sinkApp1.Stop(Seconds(simTime));

    dataRateRand->SetAttribute("Min", DoubleValue(35.0));
    dataRateRand->SetAttribute("Max", DoubleValue(65.0));  // ✅ Wider: 35-65 Mbps
    double embbRate = dataRateRand->GetValue();

    // ✅ RANDOMIZE ON/OFF DISTRIBUTION PARAMETERS TOO!
    Ptr<UniformRandomVariable> paretoScaleRand = CreateObject<UniformRandomVariable>();
    paretoScaleRand->SetAttribute("Min", DoubleValue(4.0));
    paretoScaleRand->SetAttribute("Max", DoubleValue(7.0));
    double onScale = paretoScaleRand->GetValue();
    double offScale = paretoScaleRand->GetValue() / 5.0;  // Smaller off periods

    std::ostringstream onTimeStr, offTimeStr;
    onTimeStr << "ns3::ParetoRandomVariable[Scale=" << onScale << "|Shape=1.2]";
    offTimeStr << "ns3::ParetoRandomVariable[Scale=" << offScale << "|Shape=1.5]";

    OnOffHelper onoffEmbb1("ns3::TcpSocketFactory", sinkAddr1);
    onoffEmbb1.SetAttribute("DataRate", StringValue(std::to_string((int)embbRate) + "Mbps"));
    onoffEmbb1.SetAttribute("PacketSize", UintegerValue(1400));
    onoffEmbb1.SetAttribute("OnTime", StringValue(onTimeStr.str()));
    onoffEmbb1.SetAttribute("OffTime", StringValue(offTimeStr.str()));
    onoffEmbb1.SetAttribute("Tos", UintegerValue(0x88));  // ✅ FIX: Set TOS here for TCP

    ApplicationContainer onoffEmbbApp1 = onoffEmbb1.Install(ukLAN_eMBB.Get(0));

    // ❌ REMOVED: Manual socket TOS setting that causes crash
    // Ptr<Socket> socket2 = onoffEmbbApp1.Get(0)->GetObject<OnOffApplication>()->GetSocket();
    // socket2->SetIpTos(0x88);

    onoffEmbbApp1.Start(Seconds(startTimeRand->GetValue() + 0.5));  // ✅ Staggered start
    onoffEmbbApp1.Stop(Seconds(simTime));

    RegisterTcpFlow(onoffEmbbApp1, 5001, "UK→India eMBB TCP Flow");

    std::cout << "  ✓ UK eMBB → India eMBB (TCP, " << (int)embbRate 
            << " Mbps, On=" << onScale << "s)\n";

    // 3. India eMBB → Cloud (Video streaming - RANDOMIZED RATE + BURSTINESS)
    uint16_t videoPort = 8080;
    Address cloudAddr(InetSocketAddress(backboneCloudInterfaces.GetAddress(1), videoPort));
    PacketSinkHelper videoSink("ns3::UdpSocketFactory", cloudAddr);
    ApplicationContainer videoSinkApp = videoSink.Install(cloudServices.Get(0));
    videoSinkApp.Start(Seconds(1.0));
    videoSinkApp.Stop(Seconds(simTime));

    dataRateRand->SetAttribute("Min", DoubleValue(25.0));
    dataRateRand->SetAttribute("Max", DoubleValue(45.0));  // ✅ 25-45 Mbps (wider)
    double videoRate = dataRateRand->GetValue();

    paretoScaleRand->SetAttribute("Min", DoubleValue(8.0));
    paretoScaleRand->SetAttribute("Max", DoubleValue(12.0));
    onScale = paretoScaleRand->GetValue();
    offScale = 0.3 + (paretoScaleRand->GetValue() / 20.0);  // Variable bursts

    onTimeStr.str("");
    offTimeStr.str("");
    onTimeStr << "ns3::ParetoRandomVariable[Scale=" << onScale << "|Shape=1.2]";
    offTimeStr << "ns3::ParetoRandomVariable[Scale=" << offScale << "|Shape=1.8]";

    OnOffHelper videoStream("ns3::UdpSocketFactory", cloudAddr);
    videoStream.SetAttribute("DataRate", StringValue(std::to_string((int)videoRate) + "Mbps"));
    videoStream.SetAttribute("PacketSize", UintegerValue(1400));
    videoStream.SetAttribute("OnTime", StringValue(onTimeStr.str()));
    videoStream.SetAttribute("OffTime", StringValue(offTimeStr.str()));
    videoStream.SetAttribute("Tos", UintegerValue(0x88));  // ✅ eMBB TOS

    ApplicationContainer videoApp = videoStream.Install(indiaLAN_eMBB.Get(0));

    // ❌ REMOVED: Manual socket TOS setting
    // Ptr<Socket> socket3 = videoApp.Get(0)->GetObject<OnOffApplication>()->GetSocket();
    // socket3->SetIpTos(0x88);

    videoApp.Start(Seconds(startTimeRand->GetValue() + 2.5));  // ✅ More varied start
    videoApp.Stop(Seconds(simTime));
    RegisterUdpFlow(videoApp, videoPort, "India eMBB to Cloud Video Stream");

    std::cout << "  1. ✓ India eMBB → Cloud (Video, " << (int)videoRate << " Mbps)\n";

    // 4. UK eMBB → Cloud (Video 2 - Different pattern)
    uint16_t videoPort2 = 555;
    Address cloudAddr2(InetSocketAddress(backboneCloudInterfaces.GetAddress(1), videoPort2));
    PacketSinkHelper videoSink2("ns3::UdpSocketFactory", cloudAddr2);
    ApplicationContainer videoSinkApp2 = videoSink2.Install(cloudServices.Get(0));
    videoSinkApp2.Start(Seconds(1.0));
    videoSinkApp2.Stop(Seconds(simTime));

    dataRateRand->SetAttribute("Min", DoubleValue(45.0));
    dataRateRand->SetAttribute("Max", DoubleValue(75.0));  // ✅ Higher variance
    double videoRate2 = dataRateRand->GetValue();

    onScale = 7.0 + paretoScaleRand->GetValue() / 2.0;
    offScale = 0.4 + paretoScaleRand->GetValue() / 15.0;

    onTimeStr.str("");
    offTimeStr.str("");
    onTimeStr << "ns3::ParetoRandomVariable[Scale=" << onScale << "|Shape=1.3]";
    offTimeStr << "ns3::ParetoRandomVariable[Scale=" << offScale << "|Shape=1.7]";

    OnOffHelper videoStream2("ns3::UdpSocketFactory", cloudAddr2);
    videoStream2.SetAttribute("DataRate", StringValue(std::to_string((int)videoRate2) + "Mbps"));
    videoStream2.SetAttribute("PacketSize", UintegerValue(1400));
    videoStream2.SetAttribute("OnTime", StringValue(onTimeStr.str()));
    videoStream2.SetAttribute("OffTime", StringValue(offTimeStr.str()));
    videoStream2.SetAttribute("Tos", UintegerValue(0x88));  // ✅ eMBB TOS

    ApplicationContainer videoApp2 = videoStream2.Install(ukLAN_eMBB.Get(0));

    // ❌ REMOVED: Manual socket TOS setting
    // Ptr<Socket> socket4 = videoApp2.Get(0)->GetObject<OnOffApplication>()->GetSocket();
    // socket4->SetIpTos(0x88);

    videoApp2.Start(Seconds(startTimeRand->GetValue() + 3.5));  // ✅ Even more staggered
    videoApp2.Stop(Seconds(simTime));
    RegisterUdpFlow(videoApp2, videoPort2, "UK eMBB to Cloud Video Stream");
    std::cout << "  2. ✓ UK eMBB → Cloud (Video, " << (int)videoRate2 << " Mbps)\n";

    // 5. India eMBB → UK eMBB (Video 3 - Cross-region)
    uint16_t videoPort3 = 556;
    Address ukAddr(InetSocketAddress(ukLANInterfaces.GetAddress(0), videoPort3));
    PacketSinkHelper videoSink3("ns3::UdpSocketFactory", ukAddr);
    ApplicationContainer videoSinkApp3 = videoSink3.Install(ukLAN_eMBB.Get(0));
    videoSinkApp3.Start(Seconds(1.0));
    videoSinkApp3.Stop(Seconds(simTime));

    dataRateRand->SetAttribute("Min", DoubleValue(35.0));
    dataRateRand->SetAttribute("Max", DoubleValue(65.0));
    double videoRate3 = dataRateRand->GetValue();

    onScale = 9.0 + paretoScaleRand->GetValue();
    offScale = 0.5 + paretoScaleRand->GetValue() / 10.0;

    onTimeStr.str("");
    offTimeStr.str("");
    onTimeStr << "ns3::ParetoRandomVariable[Scale=" << onScale << "|Shape=1.3]";
    offTimeStr << "ns3::ParetoRandomVariable[Scale=" << offScale << "|Shape=1.6]";

    OnOffHelper videoStream3("ns3::UdpSocketFactory", ukAddr);
    videoStream3.SetAttribute("DataRate", StringValue(std::to_string((int)videoRate3) + "Mbps"));
    videoStream3.SetAttribute("PacketSize", UintegerValue(1400));
    videoStream3.SetAttribute("OnTime", StringValue(onTimeStr.str()));
    videoStream3.SetAttribute("OffTime", StringValue(offTimeStr.str()));
    videoStream3.SetAttribute("Tos", UintegerValue(0x88));  // ✅ eMBB TOS

    ApplicationContainer videoApp3 = videoStream3.Install(indiaLAN_eMBB.Get(0));

    // ❌ REMOVED: Manual socket TOS setting
    // Ptr<Socket> socket5 = videoApp3.Get(0)->GetObject<OnOffApplication>()->GetSocket();
    // socket5->SetIpTos(0x88);

    videoApp3.Start(Seconds(startTimeRand->GetValue() + 5.0));  // ✅ Latest start
    videoApp3.Stop(Seconds(simTime));
    RegisterUdpFlow(videoApp3, videoPort3, "India eMBB to UK eMBB Video Stream");
    std::cout << "  3. ✓ India eMBB → UK eMBB (Video, " << (int)videoRate3 << " Mbps)\n\n";
    // ========== URLLC TRAFFIC (Ultra-Low Latency) ==========
    std::cout << "⚡ URLLC Traffic (Ultra-Reliable Low-Latency):\n";

    // 6. India URLLC → UK URLLC (RANDOMIZED INTERVAL)
    uint16_t urllcPort1 = 10;
    UdpEchoServerHelper urllcServer1(urllcPort1);
    ApplicationContainer urllcServerApp1 = urllcServer1.Install(ukLAN_URLLC.Get(0));
    urllcServerApp1.Start(Seconds(1.0));
    urllcServerApp1.Stop(Seconds(simTime));

    intervalRand->SetAttribute("Min", DoubleValue(12.0));
    intervalRand->SetAttribute("Max", DoubleValue(28.0));  // ✅ 12-28ms (wider for variability)
    uint32_t urllcInterval = (uint32_t)intervalRand->GetValue();

    UdpEchoClientHelper urllcClient1(ukLANInterfaces.GetAddress(1), urllcPort1);
    urllcClient1.SetAttribute("MaxPackets", UintegerValue(5000));
    urllcClient1.SetAttribute("Interval", TimeValue(MilliSeconds(urllcInterval)));
    urllcClient1.SetAttribute("PacketSize", UintegerValue(128));
    urllcClient1.SetAttribute("Tos", UintegerValue(0xb8));  // Set TOS here
    ApplicationContainer urllcClientApp1 = urllcClient1.Install(indiaLAN_URLLC.Get(0));

    urllcClientApp1.Start(Seconds(startTimeRand->GetValue()));
    urllcClientApp1.Stop(Seconds(simTime));
    std::cout << "  ✓ India URLLC → UK URLLC (interval=" << urllcInterval << "ms)\n";

    // 7. UK URLLC → India URLLC (Reverse - Different pattern) - FIXED
    uint16_t urllcPort2 = 11;
    UdpEchoServerHelper urllcServer2(urllcPort2);
    ApplicationContainer urllcServerApp2 = urllcServer2.Install(indiaLAN_URLLC.Get(0));
    urllcServerApp2.Start(Seconds(1.0));
    urllcServerApp2.Stop(Seconds(simTime));

    intervalRand->SetAttribute("Min", DoubleValue(6.0));
    intervalRand->SetAttribute("Max", DoubleValue(16.0));  // ✅ 6-16ms (faster, more varied)
    uint32_t urllcInterval2 = (uint32_t)intervalRand->GetValue();

    UdpEchoClientHelper urllcClient2(indiaLANInterfaces.GetAddress(1), urllcPort2);
    urllcClient2.SetAttribute("MaxPackets", UintegerValue(5000));
    urllcClient2.SetAttribute("Interval", TimeValue(MilliSeconds(urllcInterval2)));
    urllcClient2.SetAttribute("PacketSize", UintegerValue(64));
    urllcClient2.SetAttribute("Tos", UintegerValue(0xb8));  // ✅ FIX: Set TOS here instead of manual socket
    ApplicationContainer urllcClientApp2 = urllcClient2.Install(ukLAN_URLLC.Get(0));

    // ❌ REMOVED: Manual socket creation that causes crash
    // Ptr<Socket> socket7 = Socket::CreateSocket(ukLAN_URLLC.Get(0), UdpSocketFactory::GetTypeId());
    // socket7->SetIpTos(0xb8);  // Expedited Forwarding (EF) for URLLC

    urllcClientApp2.Start(Seconds(startTimeRand->GetValue() + 0.7));  // ✅ Slightly offset
    urllcClientApp2.Stop(Seconds(simTime));
    std::cout << "  ✓ UK URLLC → India URLLC (interval=" << urllcInterval2 << "ms, TOS=0xb8)\n";

    // 8. India URLLC → Cloud (Control signals)
    uint16_t urllcCloudPort = 12;
    Address urllcCloudAddr(InetSocketAddress(backboneCloudInterfaces.GetAddress(1), urllcCloudPort));
    PacketSinkHelper urllcCloudSink("ns3::UdpSocketFactory", urllcCloudAddr);
    ApplicationContainer urllcCloudSinkApp = urllcCloudSink.Install(cloudServices.Get(0));
    urllcCloudSinkApp.Start(Seconds(1.0));
    urllcCloudSinkApp.Stop(Seconds(simTime));

    dataRateRand->SetAttribute("Min", DoubleValue(450.0));
    dataRateRand->SetAttribute("Max", DoubleValue(750.0));  // ✅ 450-750 Kbps
    double urllcCloudRate = dataRateRand->GetValue();

    OnOffHelper urllcCloud("ns3::UdpSocketFactory", urllcCloudAddr);
    urllcCloud.SetAttribute("DataRate", StringValue(std::to_string((int)urllcCloudRate) + "Kbps"));
    urllcCloud.SetAttribute("PacketSize", UintegerValue(256));
    urllcCloud.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=2.0]"));
    urllcCloud.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.5]"));
    urllcCloud.SetAttribute("Tos", UintegerValue(0xb8));  // Set TOS here
    ApplicationContainer urllcCloudApp = urllcCloud.Install(indiaLAN_URLLC.Get(0));

    // // Set IP TOS for URLLC traffic
    // Ptr<Socket> socket8 = urllcCloudApp.Get(0)->GetObject<OnOffApplication>()->GetSocket();
    // socket8->SetIpTos(0xb8);  // Expedited Forwarding (EF) for URLLC

    urllcCloudApp.Start(Seconds(startTimeRand->GetValue() + 1.8));
    urllcCloudApp.Stop(Seconds(simTime));
    RegisterUdpFlow(urllcCloudApp, urllcCloudPort, "India URLLC to Cloud Control Signals");
    std::cout << "  ✓ India URLLC → Cloud (" << (int)urllcCloudRate << " Kbps)\n\n";

    // ========== mMTC TRAFFIC (Massive IoT) - HIGHLY VARIABLE ==========
    std::cout << "🌐 mMTC Traffic (Massive Machine-Type Communications):\n";

    uint32_t numIoTDevices = (uint32_t)iotCountRand->GetValue();
    std::cout << "  📊 Spawning " << numIoTDevices << " IoT sensors (India)\n";

    Ptr<UniformRandomVariable> iotStartRand = CreateObject<UniformRandomVariable>();
    iotStartRand->SetAttribute("Min", DoubleValue(0.0));
    iotStartRand->SetAttribute("Max", DoubleValue(10.0));  // ✅ Spread over 10 seconds!

    for (uint32_t i = 0; i < numIoTDevices; ++i) {
        uint16_t iotPort = 8000 + i;
        Address iotAddr(InetSocketAddress(backboneCloudInterfaces.GetAddress(1), iotPort));
        PacketSinkHelper iotSink("ns3::UdpSocketFactory", iotAddr);
        ApplicationContainer iotSinkApp = iotSink.Install(cloudServices.Get(0));
        iotSinkApp.Start(Seconds(1.0));
        iotSinkApp.Stop(Seconds(simTime));

        dataRateRand->SetAttribute("Min", DoubleValue(35.0));
        dataRateRand->SetAttribute("Max", DoubleValue(85.0));  // ✅ 35-85 Kbps (wider!)
        double iotRate = dataRateRand->GetValue();

        paretoScaleRand->SetAttribute("Min", DoubleValue(3.0));
        paretoScaleRand->SetAttribute("Max", DoubleValue(12.0));  // ✅ Variable ON times
        double onInterval = paretoScaleRand->GetValue();
        
        paretoScaleRand->SetAttribute("Min", DoubleValue(0.5));
        paretoScaleRand->SetAttribute("Max", DoubleValue(3.5));  // ✅ Variable OFF times
        double offInterval = paretoScaleRand->GetValue();

        onTimeStr.str("");
        offTimeStr.str("");
        onTimeStr << "ns3::ParetoRandomVariable[Scale=" << onInterval << "|Shape=1.5]";
        offTimeStr << "ns3::ParetoRandomVariable[Scale=" << offInterval << "|Shape=2.0]";

        OnOffHelper iotDevice("ns3::UdpSocketFactory", iotAddr);
        iotDevice.SetAttribute("DataRate", StringValue(std::to_string((int)iotRate) + "Kbps"));
        iotDevice.SetAttribute("PacketSize", UintegerValue(128));
        iotDevice.SetAttribute("OnTime", StringValue(onTimeStr.str()));
        iotDevice.SetAttribute("OffTime", StringValue(offTimeStr.str()));
        iotDevice.SetAttribute("Tos", UintegerValue(0x00));  // ✅ mMTC = Best Effort
        ApplicationContainer iotApp = iotDevice.Install(indiaLAN_mMTC.Get(0));
        
        // Set IP TOS for mMTC traffic (Best Effort)
        // Ptr<Socket> socketIoT = iotApp.Get(0)->GetObject<OnOffApplication>()->GetSocket();
        // socketIoT->SetIpTos(0x00);  // Best Effort for mMTC
        
        iotApp.Start(Seconds(startTimeRand->GetValue() + iotStartRand->GetValue()));  // ✅ WIDELY spread starts!
        iotApp.Stop(Seconds(simTime));
        RegisterUdpFlow(iotApp, iotPort, "India mMTC Sensor ");
    }

    std::cout << "  ✓ " << numIoTDevices << " IoT sensors: India mMTC → Cloud (spread 0-10s)\n";

    // UK mMTC (also highly randomized)
    uint32_t numUKSensors = 5 + (rand() % 6);
    std::cout << "  📊 Spawning " << numUKSensors << " sensors (UK)\n";

    for (uint32_t i = 0; i < numUKSensors; ++i) {
        uint16_t sensorPort = 8100 + i;
        Address sensorAddr(InetSocketAddress(backboneCloudInterfaces.GetAddress(1), sensorPort));
        PacketSinkHelper sensorSink("ns3::UdpSocketFactory", sensorAddr);
        ApplicationContainer sensorSinkApp = sensorSink.Install(cloudServices.Get(0));
        sensorSinkApp.Start(Seconds(1.0));
        sensorSinkApp.Stop(Seconds(simTime));

        dataRateRand->SetAttribute("Min", DoubleValue(20.0));
        dataRateRand->SetAttribute("Max", DoubleValue(50.0));  // ✅ 20-50 Kbps (wider)
        double sensorRate = dataRateRand->GetValue();

        OnOffHelper sensor("ns3::UdpSocketFactory", sensorAddr);
        sensor.SetAttribute("DataRate", StringValue(std::to_string((int)sensorRate) + "Kbps"));
        sensor.SetAttribute("PacketSize", UintegerValue(64));
        sensor.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=10.0]"));
        sensor.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=2.0]"));
        sensor.SetAttribute("Tos", UintegerValue(0x00));  // ✅ Best Effort for mMTC
        ApplicationContainer sensorApp = sensor.Install(ukLAN_mMTC.Get(0));
        
        // Set IP TOS for mMTC traffic (Best Effort)
        // Ptr<Socket> socketSensor = sensorApp.Get(0)->GetObject<OnOffApplication>()->GetSocket();
        // socketSensor->SetIpTos(0x00);  // Best Effort for mMTC
        
        sensorApp.Start(Seconds(startTimeRand->GetValue() + 0.5 + iotStartRand->GetValue()));  // ✅ Spread starts
        sensorApp.Stop(Seconds(simTime));

    }

    std::cout << "  ✓ " << numUKSensors << " sensors: UK mMTC → Cloud\n\n";

    // ========== BACKGROUND CONGESTION TRAFFIC (EARLY START + HIGH VARIANCE) ==========
    std::cout << "🔥 Background Congestion Traffic:\n";

    // ✅ CRITICAL FIX: Start background traffic EARLIER to create congestion!
    Ptr<UniformRandomVariable> bgStartRand = CreateObject<UniformRandomVariable>();
    bgStartRand->SetAttribute("Min", DoubleValue(4.0));
    bgStartRand->SetAttribute("Max", DoubleValue(8.0));  // ✅ Start at 4-8s (not 9-12s!)

    // 9. India → UK (HIGH VARIANCE)
    uint16_t bgPort1 = 9000;
    Address bgAddr1(InetSocketAddress(ukLANInterfaces.GetAddress(0), bgPort1));
    PacketSinkHelper bgSink1("ns3::TcpSocketFactory", bgAddr1);
    ApplicationContainer bgSinkApp1 = bgSink1.Install(ukLAN_eMBB.Get(0));
    bgSinkApp1.Start(Seconds(1.0));
    bgSinkApp1.Stop(Seconds(simTime));

    dataRateRand->SetAttribute("Min", DoubleValue(130.0));
    dataRateRand->SetAttribute("Max", DoubleValue(200.0));  // ✅ 130-200 Mbps (HUGE variance!)
    double bgRate1 = dataRateRand->GetValue();

    double bgStart1 = bgStartRand->GetValue();  // ✅ Early random start

    paretoScaleRand->SetAttribute("Min", DoubleValue(2.5));
    paretoScaleRand->SetAttribute("Max", DoubleValue(4.5));
    onScale = paretoScaleRand->GetValue();
    offScale = 0.8 + paretoScaleRand->GetValue() / 3.0;

    onTimeStr.str("");
    offTimeStr.str("");
    onTimeStr << "ns3::ParetoRandomVariable[Scale=" << onScale << "|Shape=1.2]";
    offTimeStr << "ns3::ParetoRandomVariable[Scale=" << offScale << "|Shape=1.5]";

    OnOffHelper bgTraffic1("ns3::TcpSocketFactory", bgAddr1);
    bgTraffic1.SetAttribute("DataRate", StringValue(std::to_string((int)bgRate1) + "Mbps"));
    bgTraffic1.SetAttribute("PacketSize", UintegerValue(1400));
    bgTraffic1.SetAttribute("OnTime", StringValue(onTimeStr.str()));
    bgTraffic1.SetAttribute("OffTime", StringValue(offTimeStr.str()));
    bgTraffic1.SetAttribute("Tos", UintegerValue(0x00));  // ✅ FIX: Set TOS here for background traffic

    ApplicationContainer bgApp1 = bgTraffic1.Install(indiaLAN_eMBB.Get(0));

    // ❌ REMOVED: Manual socket TOS setting that causes crash
    // Ptr<Socket> socketBg1 = bgApp1.Get(0)->GetObject<OnOffApplication>()->GetSocket();
    // socketBg1->SetIpTos(0x00);

    bgApp1.Start(Seconds(bgStart1));
    bgApp1.Stop(Seconds(simTime));

    RegisterTcpFlow(bgApp1, 9000, "India→UK Background TCP Flow");

    std::cout << "  ✓ Background TCP: India → UK (" << (int)bgRate1 
            << " Mbps, start=" << bgStart1 << "s)\n";

    // 10. UK → India (Reverse - Different pattern)
    uint16_t bgPort2 = 9001;
    Address bgAddr2(InetSocketAddress(indiaLANInterfaces.GetAddress(0), bgPort2));
    PacketSinkHelper bgSink2("ns3::TcpSocketFactory", bgAddr2);
    ApplicationContainer bgSinkApp2 = bgSink2.Install(indiaLAN_eMBB.Get(0));
    bgSinkApp2.Start(Seconds(1.0));
    bgSinkApp2.Stop(Seconds(simTime));

    dataRateRand->SetAttribute("Min", DoubleValue(110.0));
    dataRateRand->SetAttribute("Max", DoubleValue(180.0));  // ✅ 110-180 Mbps
    double bgRate2 = dataRateRand->GetValue();

    double bgStart2 = bgStartRand->GetValue() + 2.0;  // ✅ Offset from first background

    paretoScaleRand->SetAttribute("Min", DoubleValue(3.5));
    paretoScaleRand->SetAttribute("Max", DoubleValue(5.0));
    onScale = paretoScaleRand->GetValue();
    offScale = 1.5 + paretoScaleRand->GetValue() / 2.0;

    onTimeStr.str("");
    offTimeStr.str("");
    onTimeStr << "ns3::ExponentialRandomVariable[Mean=" << onScale << "]";
    offTimeStr << "ns3::ExponentialRandomVariable[Mean=" << offScale << "]";

    OnOffHelper bgTraffic2("ns3::TcpSocketFactory", bgAddr2);
    bgTraffic2.SetAttribute("DataRate", StringValue(std::to_string((int)bgRate2) + "Mbps"));
    bgTraffic2.SetAttribute("PacketSize", UintegerValue(1400));
    bgTraffic2.SetAttribute("OnTime", StringValue(onTimeStr.str()));
    bgTraffic2.SetAttribute("OffTime", StringValue(offTimeStr.str()));
    bgTraffic2.SetAttribute("Tos", UintegerValue(0x00));  // ✅ FIX: Set TOS here for background traffic

    ApplicationContainer bgApp2 = bgTraffic2.Install(ukLAN_eMBB.Get(0));

    // ❌ REMOVED: Manual socket TOS setting that causes crash
    // Ptr<Socket> socketBg2 = bgApp2.Get(0)->GetObject<OnOffApplication>()->GetSocket();
    // socketBg2->SetIpTos(0x00);

    bgApp2.Start(Seconds(bgStart2));
    bgApp2.Stop(Seconds(simTime));

    RegisterTcpFlow(bgApp2, 9001, "UK→India Background TCP Flow");

    std::cout << "  ✓ Background TCP: UK → India (" << (int)bgRate2 
            << " Mbps, start=" << bgStart2 << "s)\n\n";

    // ============================================================================
    // MEC CONTENT CACHING SETUP
    // ============================================================================
    std::cout << "\n💾 Installing MEC Content Caching...\n";

    // Simple HTTP cache application class
    class HttpCacheApp : public Application 
    {
    public:
        static TypeId GetTypeId() {
            static TypeId tid = TypeId("HttpCacheApp")
                .SetParent<Application>()
                .AddConstructor<HttpCacheApp>()
                .AddAttribute("CacheSize", "Maximum cache objects", 
                            UintegerValue(1000),
                            MakeUintegerAccessor(&HttpCacheApp::m_cacheSize),
                            MakeUintegerChecker<uint32_t>());
            return tid;
        }
        
        HttpCacheApp() : m_cacheSize(1000), m_hits(0), m_misses(0) {}
        
    private:
        virtual void StartApplication() {
            NS_LOG_INFO("HTTP Cache started with size: " << m_cacheSize);
        }
        
        virtual void StopApplication() {
            double hitRate = (m_hits + m_misses > 0) ? 
                            (100.0 * m_hits) / (m_hits + m_misses) : 0.0;
            NS_LOG_INFO("HTTP Cache stopped - Hit rate: " << hitRate << "%");
        }
        
        uint32_t m_cacheSize;
        uint32_t m_hits;
        uint32_t m_misses;
        std::map<std::string, Time> m_cache; // Simple cache storage
    };

    // Install cache at India MEC
    Ptr<HttpCacheApp> indiaCache = CreateObject<HttpCacheApp>();
    indiaCache->SetAttribute("CacheSize", UintegerValue(1000));
    indiaMECApps.Get(0)->AddApplication(indiaCache);
    indiaCache->SetStartTime(Seconds(0.5));
    indiaCache->SetStopTime(Seconds(simTime));

    // Install cache at UK MEC  
    Ptr<HttpCacheApp> ukCache = CreateObject<HttpCacheApp>();
    ukCache->SetAttribute("CacheSize", UintegerValue(800)); // Smaller cache for UK
    ukMECApps.Get(0)->AddApplication(ukCache);
    ukCache->SetStartTime(Seconds(0.5));
    ukCache->SetStopTime(Seconds(simTime));

    std::cout << "   ✅ India MEC: HTTP Cache (1000 objects)\n";
    std::cout << "   ✅ UK MEC: HTTP Cache (800 objects)\n";

    // ============================================================================
    // TRAFFIC SUMMARY
    // ============================================================================
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "📊 Traffic Summary (This Run):\n";
    std::cout << "  • MEC Caching: HTTP cache at both MEC nodes\n";
    std::cout << "  • Admission Control: gNodeB-level URLLC flow control\n";
    std::cout << "  • eMBB Flows: 5 (highly varied rates, starts, distributions)\n";
    std::cout << "  • URLLC Flows: 3 (varied intervals: 6-28ms range)\n";
    std::cout << "  • mMTC Flows: " << (numIoTDevices + numUKSensors) 
            << " IoT devices (VARIABLE, spread over 10s)\n";
    std::cout << "  • Background: 2 (130-200 Mbps, EARLY start at 4-10s)\n";
    std::cout << "  • Total Flows: " << (5 + 3 + numIoTDevices + numUKSensors + 2) << "\n";
    std::cout << "  • Bottleneck: Router→gNodeB (200Mbps)\n";
    std::cout << "\n  🎲 RANDOMIZED:\n";
    std::cout << "     - Data rates (wide ranges)\n";
    std::cout << "     - Start times (0-10s spread)\n";
    std::cout << "     - ON/OFF distributions (Pareto scale varies)\n";
    std::cout << "     - Packet sizes (1100-1700B for eMBB)\n";
    std::cout << "     - IoT device count (5-20 per region)\n";
    std::cout << "  📈 Every run = HIGHLY different congestion pattern!\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // ============================================================================
    // FLOW ID ESTIMATION GUIDE (for RL agent reference)
    // ============================================================================
    std::cout << "\n📋 ESTIMATED FLOW IDs (verify with mec_metrics.csv after first run):\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Flow 1: India eMBB → UK eMBB (UDP Echo)\n";
    std::cout << "  Flow 2: UK eMBB → India eMBB (TCP Bulk)\n";
    std::cout << "  Flow 3: India eMBB → Cloud (Video UDP)\n";
    std::cout << "  Flow 4: UK eMBB → Cloud (Video UDP)\n";
    std::cout << "  Flow 5: India eMBB → UK eMBB (Video UDP)\n";
    std::cout << "  Flow 6-8: URLLC flows\n";
    std::cout << "  Flow 9-N: mMTC IoT devices\n";
    std::cout << "  Flow N+1, N+2: Background TCP flows\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "⚠️  Run simulation once and check mec_metrics.csv to verify Flow IDs!\n\n";

    // ============================================================================
    // FLOW MONITOR INSTALLATION
    // ============================================================================
    std::cout << "📈 Installing FlowMonitor...\n";
    g_monitor = g_flowmon.InstallAll();
    
    // Schedule periodic metrics collection
    Simulator::Schedule(Seconds(2.0), &CollectAndPrintMetrics);
    
    std::cout << "✓ Metrics will be exported to: " << metricsFile << "\n";
    std::cout << "✓ FlowMonitor XML will be saved to: mec_flowmon.xml\n\n";

    // // ============================================================================
    // // PCAP TRACING - ALL LINKS
    // // ============================================================================
    // std::cout << "📝 Enabling PCAP traces (all links)...\n";

    // // India RAN chain
    // p2p.EnablePcap("mec_india_router_gnodeb", indiaRouterToGNodeB, true);
    // p2p.EnablePcap("mec_india_gnodeb_qos", indiaGNodeBToQoS, true);
    // p2p.EnablePcap("mec_india_qos_mec", indiaQoSToMEC, true);
    // p2p.EnablePcap("mec_india_mec_upf", indiaMECToUPF, true);
    // p2p.EnablePcap("mec_india_upf_backbone", indiaUPFToBackbone, true);

    // // UK RAN chain
    // p2p.EnablePcap("mec_uk_router_gnodeb", ukRouterToGNodeB, true);
    // p2p.EnablePcap("mec_uk_gnodeb_qos", ukGNodeBToQoS, true);
    // p2p.EnablePcap("mec_uk_qos_mec", ukQoSToMEC, true);
    // p2p.EnablePcap("mec_uk_mec_upf", ukMECToUPF, true);
    // p2p.EnablePcap("mec_uk_upf_backbone", ukUPFToBackbone, true);

    // // Backbone and external
    // p2p.EnablePcap("mec_backbone_link", backboneLink, true);
    // p2p.EnablePcap("mec_backbone_cloud", backboneToCloud, true);
    // p2p.EnablePcap("mec_backbone_internet", backboneToInternet, true);

    // // LAN segments (CSMA)
    // csma.EnablePcap("mec_india_lan", indiaLANDevices, true);
    // csma.EnablePcap("mec_uk_lan", ukLANDevices, true);

    // std::cout << "✓ PCAP enabled for ALL network links\n";
    // std::cout << "⚠️  Warning: This will generate many large files!\n\n";

    // ============================================================================
    // RUN SIMULATION
    // ============================================================================
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           STARTING SIMULATION (" << simTime << "s)                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    // ✅ Schedule episode_end signal BEFORE simulation ends
    if (enableRL && g_rlSocket && g_rlSocket->IsReady()) {
        Simulator::Schedule(Seconds(simTime - 0.5), []() {
            std::ostringstream endMsg;
            endMsg << "{"
                << "\"type\":\"episode_end\","
                << "\"timestamp\":" << Simulator::Now().GetSeconds() << ","
                << "\"reason\":\"simulation_complete\","
                << "\"simTime\":" << Simulator::Now().GetSeconds()  // ← Add this
                << "}";
            
            g_rlSocket->SendMetrics(endMsg.str());
            std::cout << "\n✅ Sent episode_end signal at t=" 
                    << Simulator::Now().GetSeconds() << "s\n";
        });
    }
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // ============================================================================
    // POST-SIMULATION METRICS
    // ============================================================================
    std::cout << "\n\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                   SIMULATION COMPLETE                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Export FlowMonitor XML
    g_monitor->SerializeToXmlFile("mec_flowmon.xml", true, true);
    
    // Print final statistics
    std::cout << "📊 Final Statistics:\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    
    g_monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(
        g_flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = g_monitor->GetFlowStats();
    
    uint32_t totalFlows = stats.size();
    uint64_t totalTxPackets = 0, totalRxPackets = 0, totalLostPackets = 0;
    double totalThroughput = 0.0;
    
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin();
         i != stats.end(); ++i) {
        totalTxPackets += i->second.txPackets;
        totalRxPackets += i->second.rxPackets;
        totalLostPackets += (i->second.txPackets - i->second.rxPackets);
        
        if (i->second.timeLastRxPacket.GetSeconds() > i->second.timeFirstTxPacket.GetSeconds()) {
            double duration = i->second.timeLastRxPacket.GetSeconds() - 
                            i->second.timeFirstTxPacket.GetSeconds();
            totalThroughput += (i->second.rxBytes * 8.0) / (duration * 1000000.0);
        }
    }
    
    std::cout << "  Total Flows: " << totalFlows << "\n";
    std::cout << "  Total TX Packets: " << totalTxPackets << "\n";
    std::cout << "  Total RX Packets: " << totalRxPackets << "\n";
    std::cout << "  Total Lost Packets: " << totalLostPackets << "\n";
    std::cout << "  Overall Loss Rate: " << std::fixed << std::setprecision(2)
              << (totalTxPackets > 0 ? (totalLostPackets * 100.0 / totalTxPackets) : 0.0) << "%\n";
    std::cout << "  Aggregate Throughput: " << std::setprecision(2) 
              << totalThroughput << " Mbps\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
    
    std::cout << "📁 Output Files Generated:\n";
    std::cout << "  ✓ " << metricsFile << " - Per-flow metrics CSV\n";
    std::cout << "  ✓ mec_flowmon.xml - FlowMonitor detailed statistics\n";
    std::cout << "  ✓ mec_bottleneck_*.pcap - PCAP traces for bottleneck links\n\n";
    
    std::cout << "🐍 Next Steps:\n";
    std::cout << "  Run Python visualization:\n";
    std::cout << "    python3 visualize_metrics.py\n\n";

    // Cleanup
    g_metricsFile.close();

    // Close RL socket if active
    if (g_rlSocket) {
        g_rlSocket->Close();
        delete g_rlSocket;
        g_rlSocket = nullptr;
    }

    Simulator::Destroy();

    return 0;
}
